"""
Data Profiling, Preprocessing & Vector Store Ingestion.

This script:
  1. Profiles the raw Stack Overflow dataset (Questions, Answers, Tags)
  2. Preprocesses and joins Q&A pairs with quality filtering
  3. Builds a ChromaDB vector store with Gemini embeddings

Usage:
    python -m app.ingest                    # default 50K samples
    python -m app.ingest --sample-size 20000  # smaller for quick testing
    python -m app.ingest --profile-only       # just run data profiling
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# HTML / text cleaning
# ──────────────────────────────────────────────────────────────────────

def clean_html(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if pd.isna(raw_html):
        return ""
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(separator="\n")
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# Step 1 — Data Profiling
# ──────────────────────────────────────────────────────────────────────

def profile_data(dataset_dir: str) -> dict:
    """
    Profile the raw dataset and print a summary report.
    Returns a dict of key metrics for use in the README.
    """
    print("\n" + "=" * 60)
    print("  📊  DATA PROFILING REPORT")
    print("=" * 60)

    # --- Questions ---
    print("\n🔍 Loading Questions (this may take a moment)...")
    q_df = pd.read_csv(
        os.path.join(dataset_dir, "Questions.csv"),
        encoding="latin-1",
        usecols=["Id", "Score", "CreationDate", "Title", "Body"],
    )
    print(f"   Total questions: {len(q_df):,}")
    print(f"   Date range: {q_df['CreationDate'].min()} → {q_df['CreationDate'].max()}")
    print(f"   Score stats:")
    print(f"     Mean:   {q_df['Score'].mean():.1f}")
    print(f"     Median: {q_df['Score'].median():.0f}")
    print(f"     P90:    {q_df['Score'].quantile(0.90):.0f}")
    print(f"     P99:    {q_df['Score'].quantile(0.99):.0f}")
    print(f"     Max:    {q_df['Score'].max():,}")
    print(f"   Questions with score >= 3: {(q_df['Score'] >= 3).sum():,} ({(q_df['Score'] >= 3).mean()*100:.1f}%)")
    print(f"   Questions with score >= 5: {(q_df['Score'] >= 5).sum():,} ({(q_df['Score'] >= 5).mean()*100:.1f}%)")

    # --- Answers ---
    print("\n🔍 Loading Answers...")
    a_df = pd.read_csv(
        os.path.join(dataset_dir, "Answers.csv"),
        encoding="latin-1",
        usecols=["Id", "ParentId", "Score", "Body"],
    )
    print(f"   Total answers: {len(a_df):,}")
    print(f"   Score stats:")
    print(f"     Mean:   {a_df['Score'].mean():.1f}")
    print(f"     Median: {a_df['Score'].median():.0f}")
    print(f"     Max:    {a_df['Score'].max():,}")

    # Coverage
    questions_with_answers = q_df["Id"].isin(a_df["ParentId"].unique())
    pct = questions_with_answers.mean() * 100
    print(f"   Questions that have at least 1 answer: {questions_with_answers.sum():,} ({pct:.1f}%)")

    avg_answers = a_df.groupby("ParentId").size().mean()
    print(f"   Avg answers per question: {avg_answers:.1f}")

    # --- Tags ---
    print("\n🔍 Loading Tags...")
    t_df = pd.read_csv(
        os.path.join(dataset_dir, "Tags.csv"),
        encoding="latin-1",
    )
    top_tags = t_df[t_df["Tag"] != "python"]["Tag"].value_counts().head(20)
    print(f"   Total tag entries: {len(t_df):,}")
    print(f"   Unique tags: {t_df['Tag'].nunique():,}")
    print(f"   Top 20 tags (besides 'python'):")
    for tag, count in top_tags.items():
        print(f"     {tag:25s} {count:>8,}")

    # Build profile dict
    profile = {
        "total_questions": len(q_df),
        "total_answers": len(a_df),
        "date_range": f"{q_df['CreationDate'].min()} → {q_df['CreationDate'].max()}",
        "q_score_mean": round(q_df["Score"].mean(), 1),
        "q_score_median": int(q_df["Score"].median()),
        "questions_with_answers_pct": round(pct, 1),
        "avg_answers_per_question": round(avg_answers, 1),
        "unique_tags": t_df["Tag"].nunique(),
        "top_tags": top_tags.head(10).to_dict(),
    }

    print("\n" + "=" * 60)
    print("  ✅  PROFILING COMPLETE")
    print("=" * 60 + "\n")

    return profile


def save_profile(profile: dict, output_path: str):
    """Save profiling results to a text file for README reference."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("DATA PROFILING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Questions:            {profile['total_questions']:,}\n")
        f.write(f"Total Answers:              {profile['total_answers']:,}\n")
        f.write(f"Date Range:                 {profile['date_range']}\n")
        f.write(f"Question Score (mean):      {profile['q_score_mean']}\n")
        f.write(f"Question Score (median):    {profile['q_score_median']}\n")
        f.write(f"Questions with Answers:     {profile['questions_with_answers_pct']}%\n")
        f.write(f"Avg Answers per Question:   {profile['avg_answers_per_question']}\n")
        f.write(f"Unique Tags:                {profile['unique_tags']:,}\n")
        f.write(f"\nTop 10 Tags (besides python):\n")
        for tag, count in profile["top_tags"].items():
            f.write(f"  {tag:25s} {count:>8,}\n")
    print(f"📄 Profile saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Step 2 & 3 — Preprocess + Build Vector Store
# ──────────────────────────────────────────────────────────────────────

def preprocess_and_build(dataset_dir: str, sample_size: int, vectorstore_path: str):
    """
    Join Q&A, filter by quality, clean HTML, build ChromaDB vector store.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from app.config import get_settings
    from app.core.embeddings import get_embeddings

    settings = get_settings()

    # ── Load & join ──────────────────────────────────────────────────
    print("\n📥 Loading Questions...")
    q_df = pd.read_csv(
        os.path.join(dataset_dir, "Questions.csv"),
        encoding="latin-1",
        usecols=["Id", "Score", "Title", "Body"],
    )
    q_df.rename(columns={"Score": "q_score", "Body": "q_body"}, inplace=True)

    print("📥 Loading Answers...")
    a_df = pd.read_csv(
        os.path.join(dataset_dir, "Answers.csv"),
        encoding="latin-1",
        usecols=["ParentId", "Score", "Body"],
    )
    a_df.rename(columns={"Score": "a_score", "Body": "a_body"}, inplace=True)

    # Keep only the best answer per question
    print("🔗 Joining questions with best answers...")
    best_answers = a_df.sort_values("a_score", ascending=False).drop_duplicates(subset="ParentId", keep="first")
    merged = q_df.merge(best_answers, left_on="Id", right_on="ParentId", how="inner")

    # Quality filter
    print(f"🔍 Before filtering: {len(merged):,} Q&A pairs")
    merged = merged[(merged["q_score"] >= 1) & (merged["a_score"] >= 1)]
    print(f"   After quality filter (q_score>=1, a_score>=1): {len(merged):,}")

    # Sort by combined score, take top N
    merged["combined_score"] = merged["q_score"] + merged["a_score"]
    merged = merged.sort_values("combined_score", ascending=False).head(sample_size)
    print(f"   Taking top {len(merged):,} by combined score")

    # ── Load tags for metadata ───────────────────────────────────────
    print("📥 Loading Tags...")
    t_df = pd.read_csv(os.path.join(dataset_dir, "Tags.csv"), encoding="latin-1")
    t_df["Tag"] = t_df["Tag"].fillna("").astype(str)
    tags_grouped = t_df.groupby("Id")["Tag"].apply(lambda x: ", ".join(x)).to_dict()

    # ── Clean HTML & create documents ────────────────────────────────
    print("🧹 Cleaning HTML and creating documents...")
    documents = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Processing"):
        q_clean = clean_html(row["q_body"])
        a_clean = clean_html(row["a_body"])
        title = str(row["Title"]).strip()

        content = f"Question: {title}\n\n{q_clean}\n\nAnswer:\n{a_clean}"

        tags = tags_grouped.get(row["Id"], "python")
        metadata = {
            "question_id": int(row["Id"]),
            "title": title[:200],  # cap for metadata storage
            "q_score": int(row["q_score"]),
            "a_score": int(row["a_score"]),
            "tags": tags[:200],
        }

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"📄 Created {len(documents):,} documents")

    # ── Chunk documents ──────────────────────────────────────────────
    print("✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\nAnswer:\n", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"   {len(documents):,} docs → {len(chunks):,} chunks")

    # ── Embed & persist ──────────────────────────────────────────────
    print(f"\n🧠 Building vector store at {vectorstore_path}")
    print(f"   Embedding model: all-MiniLM-L6-v2 (Local HuggingFace)")
    print(f"   This will take a while for {len(chunks):,} chunks...\n")

    embeddings = get_embeddings()

    # Build in batches to show progress (no API limits)
    batch_size = 500
    vectorstore = None

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i : i + batch_size]

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=vectorstore_path,
                collection_name=settings.COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"\n✅ Vector store built successfully!")
    print(f"   Location: {vectorstore_path}")
    print(f"   Total chunks indexed: {len(chunks):,}")


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest Stack Overflow data into vector store")
    parser.add_argument("--dataset-dir", default="./dataset", help="Path to dataset directory")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of Q&A pairs to index")
    parser.add_argument("--vectorstore-path", default="./vectorstore", help="Where to save the vector store")
    parser.add_argument("--profile-only", action="store_true", help="Only run data profiling, skip ingestion")
    args = parser.parse_args()

    start = time.time()

    # Step 1: Profile
    profile = profile_data(args.dataset_dir)
    save_profile(profile, "data_profile.txt")

    if args.profile_only:
        print("Profile-only mode. Exiting.")
        return

    # Step 2 & 3: Preprocess + build vector store
    preprocess_and_build(args.dataset_dir, args.sample_size, args.vectorstore_path)

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
