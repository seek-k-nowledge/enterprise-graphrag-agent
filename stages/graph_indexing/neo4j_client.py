"""
Neo4j driver wrapper for Stage 2: Graph Indexing.

Provides connection pooling, session management, query execution with retries,
and transaction support. The public API is:

  client = Neo4jClient(uri, user, password)
  client.connect()

  # Read
  records = client.query("MATCH (n) RETURN n LIMIT 10")

  # Write
  with client.write_transaction() as tx:
      result = tx.run("CREATE (n:Entity {id: $id})", id="test:1")

  # Batch
  client.batch_execute(queries)

  client.close()
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional, Iterator
from dataclasses import dataclass

from neo4j import Driver, GraphDatabase, Session, Transaction, Result
from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Wrapper around Neo4j query result for convenience."""

    records: list[dict]
    affected_rows: int = 0
    summary: Optional[Any] = None

    @staticmethod
    def from_neo4j_result(result: Result) -> "QueryResult":
        """Convert Neo4j Result to our QueryResult."""
        records = [dict(record) for record in result]
        summary = result.consume()
        affected = 0
        if summary.counters:
            # Sum all mutation counts
            affected = (
                summary.counters.nodes_created
                + summary.counters.nodes_deleted
                + summary.counters.properties_set
                + summary.counters.relationships_created
                + summary.counters.relationships_deleted
            )
        return QueryResult(records=records, affected_rows=affected, summary=summary)


class Neo4jClient:
    """
    Neo4j driver wrapper with connection pooling, error handling, and transaction support.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        timeout: int = 30,
        max_retries: int = 3,
        encrypted: bool = False,
    ):
        """
        Initialize the Neo4j client.

        Args:
            uri: Bolt connection string (e.g., bolt://localhost:7687)
            user: Username
            password: Password
            timeout: Query timeout in seconds
            max_retries: Number of retries on transient errors
            encrypted: Use encrypted connection (TLS)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.timeout = timeout
        self.max_retries = max_retries
        self.encrypted = encrypted
        self.driver: Optional[Driver] = None

    def connect(self) -> None:
        """Establish Neo4j driver connection."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                encrypted=self.encrypted,
                connection_timeout=self.timeout,
            )
            # Test the connection
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close the driver and release resources."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    def is_connected(self) -> bool:
        """Check if the driver is connected and Neo4j is reachable."""
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def health_check(self) -> dict:
        """
        Perform a health check and return Neo4j version and other info.

        Returns:
            dict with keys: connected (bool), version (str), error (str if failed)
        """
        if not self.is_connected():
            return {"connected": False, "version": None, "error": "Driver not connected"}

        try:
            with self.driver.session() as session:
                result = session.run("RETURN apoc.util.sha1($text) as hash", text="test")
                record = result.single()
                if record:
                    result2 = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
                    for record in result2:
                        if record["name"] == "Neo4j Kernel":
                            return {
                                "connected": True,
                                "version": record["versions"][0] if record["versions"] else "unknown",
                            }
                    return {"connected": True, "version": "5.x"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"connected": False, "version": None, "error": str(e)}

        return {"connected": True, "version": "unknown"}

    @contextmanager
    def session(self, read_only: bool = False) -> Iterator[Session]:
        """
        Context manager for Neo4j sessions.

        Args:
            read_only: If True, route to read replicas

        Yields:
            Neo4j Session
        """
        if not self.driver:
            raise RuntimeError("Driver not connected. Call connect() first.")

        access_mode = "READ" if read_only else "WRITE"
        session = self.driver.session(default_access_mode=access_mode)
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def read_transaction(self) -> Iterator[Transaction]:
        """Context manager for read transactions."""
        with self.session(read_only=True) as session:
            with session.begin_transaction() as tx:
                yield tx

    @contextmanager
    def write_transaction(self) -> Iterator[Transaction]:
        """Context manager for write transactions."""
        with self.session(read_only=False) as session:
            with session.begin_transaction() as tx:
                yield tx

    def query(
        self,
        cypher: str,
        parameters: Optional[dict] = None,
        read_only: bool = True,
    ) -> QueryResult:
        """
        Execute a single query.

        Args:
            cypher: Cypher query string
            parameters: Query parameters (dict)
            read_only: If True, use a read session

        Returns:
            QueryResult with records and affected row count

        Raises:
            Neo4jError on query failure after retries
        """
        if parameters is None:
            parameters = {}

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if read_only:
                    with self.session(read_only=True) as session:
                        result = session.run(cypher, parameters)
                        return QueryResult.from_neo4j_result(result)
                else:
                    with self.write_transaction() as tx:
                        result = tx.run(cypher, parameters)
                        return QueryResult.from_neo4j_result(result)
            except TransientError as e:
                last_error = e
                logger.warning(
                    f"Transient error on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt == self.max_retries:
                    raise
            except Neo4jError as e:
                logger.error(f"Neo4j error (not retryable): {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error executing query: {e}")
                raise

        if last_error:
            raise last_error

    def batch_execute(
        self,
        queries: list[tuple[str, Optional[dict]]],
        read_only: bool = False,
    ) -> list[QueryResult]:
        """
        Execute a batch of queries in a single transaction.

        Each query is a tuple of (cypher, parameters).

        Args:
            queries: List of (cypher, parameters) tuples
            read_only: If True, use a read-only transaction

        Returns:
            List of QueryResult objects, one per query

        Raises:
            Neo4jError on batch failure
        """
        if not queries:
            return []

        results = []
        context_mgr = self.read_transaction if read_only else self.write_transaction

        try:
            with context_mgr() as tx:
                for cypher, params in queries:
                    if params is None:
                        params = {}
                    result = tx.run(cypher, params)
                    results.append(QueryResult.from_neo4j_result(result))
                # Explicit commit happens when context exits
            logger.info(f"Batch executed: {len(queries)} queries")
            return results
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise

    def create_constraints(self, constraints: list[str]) -> None:
        """
        Create Neo4j constraints (idempotent).

        Each constraint is a full Cypher CREATE CONSTRAINT statement.
        Neo4j 5.x auto-ignores existing constraints.

        Args:
            constraints: List of Cypher statements
        """
        for constraint_cypher in constraints:
            try:
                self.query(constraint_cypher, read_only=False)
                logger.info(f"Constraint created or already exists: {constraint_cypher[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint creation failed (may already exist): {e}")

    def create_indexes(self, indexes: list[str]) -> None:
        """
        Create Neo4j indexes (idempotent).

        Each index is a full Cypher CREATE INDEX statement.
        Neo4j 5.x auto-ignores existing indexes.

        Args:
            indexes: List of Cypher statements
        """
        for index_cypher in indexes:
            try:
                self.query(index_cypher, read_only=False)
                logger.info(f"Index created or already exists: {index_cypher[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")

    def run_schema_setup(self, constraints: list[str], indexes: list[str]) -> None:
        """
        Run all schema setup (constraints and indexes) idempotently.

        Args:
            constraints: List of constraint Cypher statements
            indexes: List of index Cypher statements
        """
        logger.info("Setting up graph schema (constraints and indexes)...")
        self.create_constraints(constraints)
        self.create_indexes(indexes)
        logger.info("Schema setup complete")

    def merge_node(
        self,
        label: str,
        merge_key: dict,
        on_create: Optional[dict] = None,
        on_update: Optional[dict] = None,
    ) -> dict:
        """
        MERGE a node with the given label and properties.

        Utility for idempotent node creation/update.

        Args:
            label: Node label (e.g., "Entity")
            merge_key: Dict of properties to match on (becomes the ON MATCH clause)
            on_create: Properties to set on creation (optional)
            on_update: Properties to set on update (optional)

        Returns:
            Dict with keys: 'created' (bool), 'node' (dict)
        """
        if on_create is None:
            on_create = {}
        if on_update is None:
            on_update = {}

        # Build the MERGE clause
        merge_props = ", ".join(f"{k}: ${k}" for k in merge_key.keys())
        cypher = f"MERGE (n:{label} {{{merge_props}}})"

        # ON CREATE SET
        if on_create:
            on_create_props = ", ".join(f"n.{k} = ${k}_create" for k in on_create.keys())
            cypher += f" ON CREATE SET {on_create_props}"

        # ON MATCH SET
        if on_update:
            on_update_props = ", ".join(
                f"n.{k} = ${k}_update" for k in on_update.keys()
            )
            cypher += f" ON MATCH SET {on_update_props}"

        cypher += " RETURN n, elementId(n) as node_id"

        # Flatten parameters
        params = dict(merge_key)
        if on_create:
            params.update({f"{k}_create": v for k, v in on_create.items()})
        if on_update:
            params.update({f"{k}_update": v for k, v in on_update.items()})

        result = self.query(cypher, parameters=params, read_only=False)
        if result.records:
            record = result.records[0]
            return {
                "created": result.affected_rows > 0,
                "node": record.get("n"),
                "node_id": record.get("node_id"),
            }
        return {"created": False, "node": None, "node_id": None}

    def merge_relationship(
        self,
        source_label: str,
        source_key: dict,
        target_label: str,
        target_key: dict,
        relation_type: str,
        rel_properties: Optional[dict] = None,
    ) -> dict:
        """
        MERGE a relationship between two nodes.

        Utility for idempotent edge creation/update.

        Args:
            source_label: Source node label
            source_key: Source node properties to match on
            target_label: Target node label
            target_key: Target node properties to match on
            relation_type: Relationship type (e.g., "MENTIONED_IN")
            rel_properties: Properties to set on the relationship (optional)

        Returns:
            Dict with keys: 'created' (bool), 'relationship' (dict)
        """
        if rel_properties is None:
            rel_properties = {}

        # Build MATCH and MERGE clauses
        source_props = ", ".join(f"{k}: ${k}_src" for k in source_key.keys())
        target_props = ", ".join(f"{k}: ${k}_tgt" for k in target_key.keys())

        cypher = (
            f"MATCH (src:{source_label} {{{source_props}}}), "
            f"(tgt:{target_label} {{{target_props}}})"
        )
        cypher += f" MERGE (src)-[r:{relation_type}]->(tgt)"

        if rel_properties:
            rel_props_str = ", ".join(f"r.{k} = ${k}_rel" for k in rel_properties.keys())
            cypher += f" ON CREATE SET {rel_props_str}"

        cypher += " RETURN r, elementId(r) as rel_id"

        params = {}
        for k, v in source_key.items():
            params[f"{k}_src"] = v
        for k, v in target_key.items():
            params[f"{k}_tgt"] = v
        for k, v in rel_properties.items():
            params[f"{k}_rel"] = v

        result = self.query(cypher, parameters=params, read_only=False)
        if result.records:
            record = result.records[0]
            return {
                "created": result.affected_rows > 0,
                "relationship": record.get("r"),
                "rel_id": record.get("rel_id"),
            }
        return {"created": False, "relationship": None, "rel_id": None}

    def execute_cypher(self, cypher: str, parameters: Optional[dict] = None) -> QueryResult:
        """
        Execute raw Cypher (read or write detected automatically from statement).

        Args:
            cypher: Cypher query string
            parameters: Query parameters

        Returns:
            QueryResult

        Raises:
            Neo4jError on failure
        """
        read_only = not any(
            keyword in cypher.upper() for keyword in ["CREATE", "MERGE", "SET", "DELETE"]
        )
        return self.query(cypher, parameters=parameters, read_only=read_only)
