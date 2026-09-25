"""Load a pipeline output JSON file into Neo4j as a knowledge graph.

Each triplet becomes two nodes (subject / object) connected by a typed
relationship.  Nodes are labelled ``KnowledgeGraphNode`` and carry a
``name`` property (the entity text) plus an ``entity_type`` property set
to ``"Entity"`` for all extracted entities.  Relationships carry a
``definition`` property taken from the ``relation_definitions`` map in the
JSON, so every edge is self-documenting in the graph.

The script is **idempotent**: it uses ``MERGE`` throughout, so it is safe
to run multiple times or to layer outputs from different pipeline runs onto
the same database.

Usage
-----
    # Using .env for credentials (recommended)
    python creation/without-schema/ingest_neo4j.py --input output.json

    # Explicit credentials
    python creation/without-schema/ingest_neo4j.py \\
        --input  output.json            \\
        --uri    bolt://localhost:7687  \\
        --user   neo4j                 \\
        --password secret

Credentials are read from (in priority order):
    1. CLI flags  --uri / --user / --password
    2. Environment variables  NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
    3. NEO4J_AUTH=user/password  (Docker convenience variable)
    4. Defaults: bolt://localhost:7687 / neo4j
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

# Load project .env (two levels up: project/creation/without-schema/)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

_REL_SANITIZE = re.compile(r"[^A-Za-z0-9_]")


def _to_rel_type(relation: str) -> str:
    """Convert a camelCase / arbitrary relation name to a valid Neo4j relationship type.

    Neo4j relationship types must match  [A-Za-z_][A-Za-z0-9_]*  when used
    without backtick quoting.  This function:
      - Splits camelCase words with underscores  (ceoOf → CEO_OF)
      - Replaces remaining special characters with underscores
      - Uppercases the result
      - Collapses consecutive underscores
      - Ensures it starts with a letter or underscore
    """
    # Insert underscore before each uppercase-letter run
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", relation)
    # Replace anything not alphanumeric/underscore
    sanitized = _REL_SANITIZE.sub("_", spaced)
    # Uppercase + collapse runs of underscores
    result = re.sub(r"_+", "_", sanitized).upper().strip("_")
    return result or "RELATED_TO"


def _ensure_constraint(session: Any, database: str) -> None:
    """Create a uniqueness constraint on KnowledgeGraphNode.name (idempotent)."""
    session.run(
        "CREATE CONSTRAINT kg_node_name_unique IF NOT EXISTS "
        "FOR (n:KnowledgeGraphNode) REQUIRE n.name IS UNIQUE"
    )
    logger.debug("Constraint ensured on KnowledgeGraphNode.name")


def _merge_node(session: Any, name: str) -> None:
    session.run(
        "MERGE (n:KnowledgeGraphNode {name: $name}) "
        "ON CREATE SET n.entity_type = 'Entity'",
        name=name,
    )


def _merge_relationship(
    session: Any, subject: str, rel_type: str, obj: str, definition: str
) -> None:
    # Relationship types cannot be parameterised in Cypher — they must be
    # interpolated.  _to_rel_type() sanitizes the value before interpolation.
    session.run(
        f"MATCH (s:KnowledgeGraphNode {{name: $subject}}), "
        f"      (o:KnowledgeGraphNode {{name: $obj}}) "
        f"MERGE (s)-[r:{rel_type}]->(o) "
        f"ON CREATE SET r.definition = $definition "
        f"ON MATCH  SET r.definition = $definition",
        subject=subject,
        obj=obj,
        definition=definition,
    )


# ---------------------------------------------------------------------------
# Ingest logic
# ---------------------------------------------------------------------------

def ingest(
    graph_json: dict,
    uri: str,
    username: str,
    password: str,
    database: str,
) -> tuple[int, int]:
    """Write all triplets from *graph_json* into Neo4j.

    Returns ``(nodes_merged, relationships_merged)``.
    """
    triplets = graph_json.get("triplets", [])
    relation_definitions: dict[str, str] = graph_json.get("relation_definitions", {})

    if not triplets:
        logger.warning("No triplets found in input — nothing to ingest.")
        return 0, 0

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s (database: %s)", uri, database)

        nodes_merged = 0
        rels_merged = 0

        with driver.session(database=database) as session:
            _ensure_constraint(session, database)

            for triplet in triplets:
                subject = triplet.get("subject", "").strip()
                # JSON uses "object" as the key (by_alias=True serialisation)
                obj = triplet.get("object", "").strip()
                canonical = triplet.get("canonical_relation") or triplet.get("relation", "")
                canonical = canonical.strip()

                if not subject or not obj or not canonical:
                    logger.debug("Skipping incomplete triplet: %s", triplet)
                    continue

                rel_type = _to_rel_type(canonical)
                definition = relation_definitions.get(canonical, "")

                _merge_node(session, subject)
                _merge_node(session, obj)
                nodes_merged += 2  # approximate; MERGE may match existing

                _merge_relationship(session, subject, rel_type, obj, definition)
                rels_merged += 1

    except AuthError as exc:
        raise SystemExit(f"Neo4j authentication failed: {exc}") from exc
    except ServiceUnavailable as exc:
        raise SystemExit(
            f"Cannot reach Neo4j at {uri}. Is the database running? ({exc})"
        ) from exc
    finally:
        driver.close()

    return nodes_merged, rels_merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="Path to the pipeline output JSON file (created by main.py).",
)
@click.option(
    "--uri",
    default=None,
    envvar="NEO4J_URI",
    show_default="bolt://localhost:7687",
    help="Neo4j Bolt URI.",
)
@click.option(
    "--user",
    default=None,
    envvar="NEO4J_USERNAME",
    show_default="neo4j",
    help="Neo4j username.",
)
@click.option(
    "--password",
    default=None,
    envvar="NEO4J_PASSWORD",
    help="Neo4j password (also accepts NEO4J_AUTH=user/password).",
)
@click.option(
    "--database",
    default=None,
    envvar="NEO4J_DATABASE",
    show_default="neo4j",
    help="Target Neo4j database.",
)
def main(
    input_path: Path,
    uri: str | None,
    user: str | None,
    password: str | None,
    database: str | None,
) -> None:
    """Ingest a pipeline output JSON into Neo4j.

    Reads triplets produced by the without-schema pipeline and writes them
    as nodes and relationships into the configured Neo4j database.
    Every ``MERGE`` is idempotent — safe to run multiple times.
    """
    # Resolve credentials — mirror knowledge_graph.connection_settings() logic
    import os
    resolved_uri      = uri      or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    resolved_user     = user     or os.getenv("NEO4J_USERNAME", "neo4j")
    resolved_database = database or os.getenv("NEO4J_DATABASE", "neo4j")

    resolved_password = password or os.getenv("NEO4J_PASSWORD")
    if not resolved_password:
        auth_env = os.getenv("NEO4J_AUTH", "")
        if "/" in auth_env:
            resolved_user, resolved_password = auth_env.split("/", maxsplit=1)

    if not resolved_password:
        raise click.UsageError(
            "Neo4j password is required. Set --password, NEO4J_PASSWORD, or NEO4J_AUTH in .env."
        )

    graph_json = json.loads(input_path.read_text(encoding="utf-8"))
    total_triplets = len(graph_json.get("triplets", []))
    click.echo(f"Input : {input_path}  ({total_triplets} triplet(s))", err=True)

    nodes_merged, rels_merged = ingest(
        graph_json=graph_json,
        uri=resolved_uri,
        username=resolved_user,
        password=resolved_password,
        database=resolved_database,
    )

    click.echo(
        f"Done  : ~{nodes_merged} node merge(s), {rels_merged} relationship merge(s) "
        f"into '{resolved_database}' @ {resolved_uri}",
        err=True,
    )


if __name__ == "__main__":
    main()

