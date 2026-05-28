from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import torch
from loguru import logger
from neo4j import GraphDatabase
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class NodeType(str, Enum):
    PATIENT = "Patient"
    GENE = "Gene"
    VARIANT = "Variant"
    DRUG = "Drug"
    PHENOTYPE = "Phenotype"
    DISEASE = "Disease"
    PATHWAY = "Pathway"
    PROTEIN = "Protein"


class RelationType(str, Enum):
    HAS_VARIANT = "HAS_VARIANT"
    AFFECTS_GENE = "AFFECTS_GENE"
    TREATED_WITH = "TREATED_WITH"
    RESPONDS_TO = "RESPONDS_TO"
    HAS_PHENOTYPE = "HAS_PHENOTYPE"
    CAUSES = "CAUSES"
    TARGETS = "TARGETS"
    INTERACTS_WITH = "INTERACTS_WITH"
    PART_OF_PATHWAY = "PART_OF_PATHWAY"
    ENCODES = "ENCODES"


@dataclass
class Node:
    id: str
    type: NodeType
    properties: Dict[str, Any]


@dataclass
class Relationship:
    source: str
    target: str
    type: RelationType
    properties: Dict[str, Any]


class Neo4jKnowledgeGraph:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.logger = logger.bind(component="Neo4jKnowledgeGraph")

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jKnowledgeGraph":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def create_indexes(self) -> None:
        indexes = [
            "CREATE INDEX patient_id IF NOT EXISTS FOR (p:Patient) ON (p.id)",
            "CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)",
            "CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX variant_id IF NOT EXISTS FOR (v:Variant) ON (v.id)",
        ]

        with self.driver.session(database=self.database) as session:
            for index in indexes:
                session.run(index)

        self.logger.info("Indexes created successfully")

    def create_constraints(self) -> None:
        constraints = [
            (
                "CREATE CONSTRAINT patient_unique IF NOT EXISTS "
                "FOR (p:Patient) REQUIRE p.id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT gene_unique IF NOT EXISTS "
                "FOR (g:Gene) REQUIRE g.symbol IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT drug_unique IF NOT EXISTS "
                "FOR (d:Drug) REQUIRE d.name IS UNIQUE"
            ),
        ]

        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as exc:
                    self.logger.warning(f"Constraint creation skipped: {exc}")

        self.logger.info("Constraints created successfully")

    def add_node(self, node: Node) -> None:
        query = f"""
        MERGE (n:{node.type.value} {{id: $id}})
        SET n += $properties
        RETURN n
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                id=node.id,
                properties=node.properties,
            )

    def add_relationship(self, rel: Relationship) -> None:
        query = f"""
        MATCH (a {{id: $source}})
        MATCH (b {{id: $target}})
        MERGE (a)-[r:{rel.type.value}]->(b)
        SET r += $properties
        RETURN r
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                source=rel.source,
                target=rel.target,
                properties=rel.properties,
            )

    def batch_add_nodes(self, nodes: List[Node]) -> None:
        query = """
        UNWIND $nodes AS node
        CALL {
            WITH node
            CALL apoc.merge.node(
                [node.type],
                {id: node.id},
                node.properties,
                node.properties
            ) YIELD node AS n
            RETURN n
        }
        RETURN count(*) as created
        """

        nodes_data = [
            {
                "type": node.type.value,
                "id": node.id,
                "properties": node.properties,
            }
            for node in nodes
        ]

        with self.driver.session(database=self.database) as session:
            result = session.run(query, nodes=nodes_data)
            created = result.single()["created"]
            self.logger.info(f"Created {created} nodes")

    def find_patient_subgraph(
        self,
        patient_id: str,
        depth: int = 2,
    ) -> nx.DiGraph:
        query = f"""
        MATCH path = (p:Patient {{id: $patient_id}})-[*1..{depth}]-(n)
        RETURN path
        """

        graph = nx.DiGraph()

        with self.driver.session(database=self.database) as session:
            result = session.run(query, patient_id=patient_id)

            for record in result:
                path = record["path"]

                for node in path.nodes:
                    graph.add_node(
                        node.element_id,
                        **dict(node),
                    )

                for relationship in path.relationships:
                    graph.add_edge(
                        relationship.start_node.element_id,
                        relationship.end_node.element_id,
                        type=relationship.type,
                        **dict(relationship),
                    )

        return graph

    def find_similar_patients(
        self,
        patient_id: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        query = """
        MATCH (p1:Patient {id: $patient_id})-[:HAS_VARIANT]->(v:Variant)
        MATCH (p2:Patient)-[:HAS_VARIANT]->(v)
        WHERE p1 <> p2
        WITH p1, p2, count(v) as shared_variants
        MATCH (p1)-[:HAS_PHENOTYPE]->(ph:Phenotype)
        MATCH (p2)-[:HAS_PHENOTYPE]->(ph)
        WITH p1, p2, shared_variants, count(ph) as shared_phenotypes
        WITH p2.id as similar_patient,
             (shared_variants * 2.0 + shared_phenotypes) as similarity_score
        RETURN similar_patient, similarity_score
        ORDER BY similarity_score DESC
        LIMIT $top_k
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                patient_id=patient_id,
                top_k=top_k,
            )

            return [
                (record["similar_patient"], record["similarity_score"])
                for record in result
            ]

    def find_drug_targets(
        self,
        gene_symbols: List[str],
    ) -> List[Dict[str, Any]]:
        query = """
        MATCH (g:Gene)-[:ENCODES]->(p:Protein)<-[:TARGETS]-(d:Drug)
        WHERE g.symbol IN $gene_symbols
        RETURN DISTINCT d.name as drug,
               d.mechanism as mechanism,
               collect(g.symbol) as targeted_genes,
               d.approval_status as status
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, gene_symbols=gene_symbols)

            return [dict(record) for record in result]

    def compute_centrality(self, node_type: NodeType) -> Dict[str, float]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (n:{node_type.value})
                RETURN n.id as node_id,
                       size((n)--()) as degree
                ORDER BY degree DESC
                """
            )

            return {
                record["node_id"]: float(record["degree"])
                for record in result
            }


class EpilepsyGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=4, concat=True)
        )
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * 4))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * 4,
                    hidden_channels,
                    heads=4,
                    concat=True,
                )
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * 4))

        self.convs.append(
            GATConv(
                hidden_channels * 4,
                out_channels,
                heads=1,
                concat=False,
            )
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for index, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[index](x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        return self.convs[-1](x, edge_index)


class KnowledgeGraphEmbedder:
    def __init__(
        self,
        neo4j_graph: Neo4jKnowledgeGraph,
        embedding_dim: int = 128,
    ) -> None:
        self.neo4j_graph = neo4j_graph
        self.embedding_dim = embedding_dim
        self.model: Optional[EpilepsyGNN] = None
        self.logger = logger.bind(component="KnowledgeGraphEmbedder")

    def convert_to_pyg(
        self,
        patient_ids: Optional[List[str]] = None,
    ) -> Data:
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN id(n) as source_id, labels(n) as source_labels,
               properties(n) as source_props,
               type(r) as rel_type,
               id(m) as target_id, labels(m) as target_labels,
               properties(m) as target_props
        """

        node_map = {}
        node_features = []
        edges = []

        with self.neo4j_graph.driver.session(
            database=self.neo4j_graph.database
        ) as session:
            result = session.run(query)

            for record in result:
                source_id = record["source_id"]

                if source_id not in node_map:
                    node_map[source_id] = len(node_map)
                    node_features.append(
                        self._create_node_features(
                            record["source_labels"],
                            record["source_props"],
                        )
                    )

                if record["rel_type"]:
                    target_id = record["target_id"]

                    if target_id not in node_map:
                        node_map[target_id] = len(node_map)
                        node_features.append(
                            self._create_node_features(
                                record["target_labels"],
                                record["target_props"],
                            )
                        )

                    edges.append(
                        [
                            node_map[source_id],
                            node_map[target_id],
                        ]
                    )

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def _create_node_features(
        self,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> List[float]:
        features = [0.0] * self.embedding_dim

        node_type_map = {
            "Patient": 0,
            "Gene": 1,
            "Drug": 2,
            "Variant": 3,
            "Phenotype": 4,
        }

        for label in labels:
            if label in node_type_map:
                features[node_type_map[label]] = 1.0

        return features

    def train(
        self,
        data: Data,
        epochs: int = 100,
        lr: float = 0.001,
    ) -> None:
        self.model = EpilepsyGNN(
            in_channels=data.x.size(1),
            hidden_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            output = self.model(data.x, data.edge_index)
            loss = torch.nn.functional.mse_loss(output, data.x)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}"
                )

    def get_embeddings(self, data: Data) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model has not been trained.")

        self.model.eval()

        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)

        return embeddings
