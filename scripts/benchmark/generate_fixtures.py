#!/usr/bin/env python3
"""
Generate realistic benchmark fixtures with optional embedding vectors.

Provides dynamic document generation with varied metadata structures,
nested fields, realistic text content, and optional OpenAI/Gemini embeddings.
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# ============================================================================
# CONTENT GENERATORS
# ============================================================================

TECH_TOPICS = [
    "machine learning",
    "artificial intelligence",
    "cloud computing",
    "database optimization",
    "web development",
    "data science",
    "cybersecurity",
    "mobile apps",
    "devops practices",
    "software architecture",
    "containerization",
    "microservices",
    "api design",
    "performance tuning",
    "distributed systems",
    "system design",
    "network protocols",
    "testing strategies",
    "agile methodology",
    "code optimization",
]

CATEGORIES = [
    "programming",
    "infrastructure",
    "data",
    "security",
    "design",
    "operations",
    "architecture",
    "performance",
    "testing",
    "deployment",
]

TAGS_POOL = [
    "python",
    "javascript",
    "golang",
    "rust",
    "java",
    "kubernetes",
    "docker",
    "aws",
    "gcp",
    "azure",
    "postgresql",
    "mongodb",
    "redis",
    "elasticsearch",
    "tensorflow",
    "pytorch",
    "fastapi",
    "django",
    "react",
    "vue",
    "github",
    "gitlab",
    "ci-cd",
    "monitoring",
    "logging",
    "caching",
    "indexing",
    "optimization",
    "scalability",
    "reliability",
]

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]
CONTENT_TYPES = ["tutorial", "guide", "reference", "best-practice", "case-study"]

LOREM_TEMPLATES = {
    "machine learning": [
        "Machine learning models require careful feature engineering and validation. Overfitting is a common challenge when training on limited datasets. Cross-validation and regularization techniques help improve generalization.",
        "Deep learning has revolutionized many fields including computer vision and natural language processing. Neural networks with multiple layers can learn complex patterns. Training requires substantial computational resources and careful hyperparameter tuning.",
        "Supervised learning uses labeled data to train predictive models. Classification and regression are the two main categories. Model evaluation metrics include accuracy, precision, recall, and F1-score.",
        "Unsupervised learning discovers patterns in unlabeled data. Clustering and dimensionality reduction are common techniques. K-means, hierarchical clustering, and DBSCAN are popular algorithms.",
    ],
    "cloud computing": [
        "Cloud platforms like AWS, GCP, and Azure provide scalable infrastructure. Containers and serverless computing offer flexible deployment options. Cost optimization and security are critical considerations.",
        "Infrastructure as Code (IaC) enables automated provisioning and management. Terraform and CloudFormation are popular tools. Version control and CI/CD integration ensure reliable deployments.",
        "Cloud-native applications are designed for scalability and resilience. Microservices architecture allows independent scaling of components. Load balancing and auto-scaling ensure high availability.",
        "Multi-cloud strategies reduce vendor lock-in. Kubernetes provides portable orchestration across clouds. API gateways and service meshes manage inter-service communication.",
    ],
    "database optimization": [
        "Database indexing dramatically improves query performance. B-tree and hash indexes are common types. Query optimization requires understanding execution plans and statistics.",
        "Normalization reduces data redundancy and improves consistency. However, denormalization may be necessary for performance in specific scenarios. Sharding distributes data across multiple instances.",
        "Connection pooling manages database connections efficiently. Caching layers reduce database load. Read replicas enable horizontal scaling for read-heavy workloads.",
        "Transaction management ensures data consistency. ACID properties are fundamental. Deadlock prevention and lock management are important considerations.",
    ],
    "web development": [
        "Frontend frameworks like React, Vue, and Angular provide efficient UI development. State management is crucial for complex applications. Build tools and transpilers optimize code delivery.",
        "Backend frameworks enable rapid API development. REST and GraphQL are popular architectural styles. Authentication and authorization must be carefully implemented.",
        "Web security is critical for protecting user data. XSS, SQL injection, and CSRF are common vulnerabilities. HTTPS, Content Security Policy, and secure headers provide protection.",
        "Performance optimization includes minification, compression, and caching. Code splitting reduces initial load times. Progressive enhancement improves user experience.",
    ],
}


def generate_realistic_text(topic: str = None, length: str = "medium") -> str:
    """Generate realistic benchmark text content."""
    if topic is None:
        topic = random.choice(list(LOREM_TEMPLATES.keys()))

    templates = LOREM_TEMPLATES.get(topic, LOREM_TEMPLATES["machine learning"])

    if length == "short":
        return random.choice(templates).split(".")[0] + "."
    elif length == "long":
        sentences = []
        for _ in range(random.randint(4, 5)):
            template = random.choice(templates)
            sentence = template.split(".")[random.randint(0, len(template.split(".")) - 2)] + "."
            sentences.append(sentence)
        return " ".join(sentences)
    else:  # medium
        template = random.choice(templates)
        sentences = template.split(".")[: random.randint(2, 3)]
        return ".".join(sentences) + "."


def generate_nested_metadata(index: int) -> Dict[str, Any]:
    """Generate complex nested metadata structure.

    Note: All list values are converted to comma-separated strings for
    compatibility with ChromaDB and other backends that don't support
    list metadata values.
    """
    random.seed(index)

    # Generate lists first, then convert to strings
    frameworks_list = random.sample(TAGS_POOL[:10], k=random.randint(1, 3))
    languages_list = random.sample(["python", "javascript", "golang", "rust", "java"], k=random.randint(1, 2))
    tags_list = random.sample(TAGS_POOL, k=random.randint(3, 8))

    return {
        "title": f"Document {index}: {random.choice(TECH_TOPICS)}",
        "difficulty": random.choice(DIFFICULTY_LEVELS),
        "content_type": random.choice(CONTENT_TYPES),
        "author": {
            "name": f"Author_{index % 100}",
            "expertise": random.choice(CATEGORIES),
            "contributions": random.randint(1, 50),
        },
        "publication": {
            "date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "year": random.randint(2020, 2025),
            "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 20)}",
        },
        "stats": {
            "views": random.randint(10, 100000),
            "likes": random.randint(0, 10000),
            "shares": random.randint(0, 5000),
            "comments": random.randint(0, 1000),
        },
        "technical": {
            "frameworks": ", ".join(frameworks_list),  # Convert list to comma-separated string
            "languages": ", ".join(languages_list),  # Convert list to comma-separated string
            "complexity_score": round(random.uniform(0.1, 1.0), 2),
        },
        "tags": ", ".join(tags_list),  # Convert list to comma-separated string
        "rating": round(random.uniform(1.0, 5.0), 1),
        "word_count": random.randint(100, 10000),
        "read_time_minutes": random.randint(1, 60),
    }


def generate_benchmark_docs(num_docs: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate benchmark documents with rich metadata."""
    random.seed(seed)
    docs = []

    for i in range(num_docs):
        category = random.choice(CATEGORIES)
        topic = random.choice(TECH_TOPICS)
        text_length = random.choice(["short", "medium", "long"])
        text = generate_realistic_text(topic, text_length)

        doc = {
            "id": f"doc_{seed}_{i:06d}",
            "text": text,
            "category": category,
            "score": round(random.uniform(0.0, 1.0), 3),
            "metadata": generate_nested_metadata(i),
        }
        docs.append(doc)

    return docs


def generate_search_queries(num_queries: int = 100, seed: int = 42) -> List[str]:
    """Generate diverse search queries."""
    random.seed(seed)
    queries = []

    # Single-word queries
    for _ in range(num_queries // 3):
        queries.append(random.choice(TECH_TOPICS))

    # Multi-word queries
    for _ in range(num_queries // 3):
        topic1 = random.choice(TECH_TOPICS)
        topic2 = random.choice(TECH_TOPICS)
        queries.append(f"{topic1} and {topic2}")

    # Phrase queries
    phrases = [
        "best practices for",
        "how to implement",
        "introduction to",
        "advanced techniques in",
        "guide to",
        "tutorial on",
        "comparison of",
        "optimization strategies for",
    ]
    for _ in range(num_queries - len(queries)):
        phrase = random.choice(phrases)
        topic = random.choice(TECH_TOPICS)
        queries.append(f"{phrase} {topic}")

    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Generate large-scale benchmark fixtures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default fixtures (10,000 docs, 1,000 queries)
  python generate_fixtures.py

  # Generate custom size
  python generate_fixtures.py --docs 5000 --queries 500

  # Output to custom location
  python generate_fixtures.py --output data/fixtures.json
        """,
    )

    parser.add_argument(
        "--docs",
        type=int,
        default=10000,
        help="Number of documents to generate (default: 10000)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=1000,
        help="Number of search queries to generate (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/{provider}_{docs}.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--add-vectors",
        action="store_true",
        help="Generate vectors using embedding provider (OpenAI/Gemini)",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["openai", "gemini"],
        default="openai",
        help="Embedding provider to use for vector generation (default: openai)",
    )

    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output is None:
        provider = args.embedding_provider if args.add_vectors else "static"
        # Get the script's directory to ensure correct path
        script_dir = Path(__file__).parent
        args.output = str(script_dir / "data" / f"{provider}_{args.docs}.json")

    # Generate fixtures
    print("\n" + "=" * 70)
    print("Generating Benchmark Fixtures")
    print("=" * 70)
    print(f"Documents: {args.docs:,}")
    print(f"Queries: {args.queries:,}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    if args.add_vectors:
        print(f"Embedding: {args.embedding_provider.upper()}")
    print("=" * 70 + "\n")

    print(f"Generating {args.docs:,} documents with nested metadata...")
    docs = generate_benchmark_docs(args.docs, seed=args.seed)
    print(f"Generated {len(docs):,} documents\n")

    print(f"Generating {args.queries:,} diverse search queries...")
    queries = generate_search_queries(args.queries, seed=args.seed)
    print(f"Generated {len(queries):,} queries\n")

    # Generate vectors if requested
    if args.add_vectors:
        print(f"Generating vectors using {args.embedding_provider.upper()} embedding...")

        try:
            if args.embedding_provider == "openai":
                from crossvector.embeddings.openai import OpenAIEmbeddingAdapter

                embedding_adapter = OpenAIEmbeddingAdapter(model_name="text-embedding-3-small")
            else:  # gemini
                from crossvector.embeddings.gemini import GeminiEmbeddingAdapter

                embedding_adapter = GeminiEmbeddingAdapter(model_name="gemini-embedding-001", dim=1536)

            print(f"   Model: {embedding_adapter.model_name}")
            print(f"   Dimension: {embedding_adapter.dim}")

            # Generate vectors in batches to avoid rate limits
            batch_size = 500
            total_docs = len(docs)

            for i in range(0, total_docs, batch_size):
                batch = docs[i : i + batch_size]
                batch_texts = [doc["text"] for doc in batch]

                print(
                    f"   Processing batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch)} docs)..."
                )
                vectors = embedding_adapter.get_embeddings(batch_texts)

                # Add vectors to documents
                for doc, vector in zip(batch, vectors):
                    doc["vector"] = vector

            print(f"Generated {total_docs:,} vectors using {args.embedding_provider.upper()}\n")

        except Exception as e:
            print(f"Failed to generate vectors: {e}")
            print("   Fixtures will be saved without vectors\n")

    # Calculate statistics
    total_text_length = sum(len(doc["text"]) for doc in docs)
    avg_text_length = total_text_length / len(docs)

    print("Fixture Statistics:")
    print(f"  • Total documents: {len(docs):,}")
    print(f"  • Total queries: {len(queries):,}")
    print(f"  • Avg text length: {avg_text_length:.0f} chars")
    print(f"  • Total data size: {total_text_length / 1024 / 1024:.1f} MB\n")

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "metadata": {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "total_documents": len(docs),
            "total_queries": len(queries),
            "total_text_size_bytes": total_text_length,
            "average_text_length": round(avg_text_length, 1),
            "categories": list(set(doc["category"] for doc in docs)),
            "num_categories": len(set(doc["category"] for doc in docs)),
        },
        "documents": docs,
        "queries": queries,
    }

    with open(output_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print("=" * 70)
    print("Fixtures Generated Successfully!")
    print("=" * 70)
    print(f"File: {output_path.absolute()}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Documents: {len(docs):,}")
    print(f"Queries: {len(queries):,}")
    print("=" * 70 + "\n")

    print("💡 Usage in benchmark:")
    print("  import json")
    print(f"  with open('{args.output}') as f:")
    print("      fixtures = json.load(f)")
    print("  docs = fixtures['documents']")
    print("  queries = fixtures['queries']\n")


if __name__ == "__main__":
    main()
