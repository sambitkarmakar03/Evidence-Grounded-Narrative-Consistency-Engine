import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import re
from dataclasses import dataclass
import numpy as np

# Replaces: from utils.chunking import ChunkConfig, SmartChunker, ...

@dataclass
class ChunkConfig:
    target_chunk_size: int = 1000
    min_chunk_size: int = 300
    overlap_chars: int = 150

class SingleVectorDatabase:
    def __init__(self, novel_name, embedding_model, device='cpu'):
        self.novel_name = novel_name
        self.model = SentenceTransformer(embedding_model, device=device)
        self.index = None
        self.chunks = []

    def build(self, text, config):
        # 1. Split into sentences (Smart Chunking)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            
            if current_length + len(sentence) <= config.target_chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                # Store completed chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= config.min_chunk_size:
                    self.chunks.append({
                        "text": chunk_text,
                        "length": len(chunk_text),
                        "num_sentences": len(current_chunk),
                        "num_paragraphs": chunk_text.count('\n\n') + 1
                    })
                
                # Handle overlap: Keep the last few sentences for context
                # Simple version: start new chunk with current sentence
                current_chunk = [sentence]
                current_length = len(sentence)

        # 2. Vectorize
        embeddings = self.model.encode([c['text'] for c in self.chunks], convert_to_tensor=False)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        return len(self.chunks)

    def query(self, text, top_k=5):
        query_vec = self.model.encode([text])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), top_k)
        return [self.chunks[i] for i in indices[0]]

class DualVectorDatabaseManager:
    def __init__(self, novel1_name, novel1_path, novel2_name, novel2_path, embedding_model, chunk_config):
        self.db1 = SingleVectorDatabase(novel1_name, embedding_model)
        self.db2 = SingleVectorDatabase(novel2_name, embedding_model)
        self.paths = [novel1_path, novel2_path]
        self.databases = [self.db1, self.db2]
        self.config = chunk_config

    def build_all(self):
        stats = []
        for i, db in enumerate(self.databases):
            with open(self.paths[i], 'r', encoding='utf-8') as f:
                content = f.read()
            num_chunks = db.build(content, self.config)
            stats.append({
                "num_chunks": num_chunks,
                "total_chars": len(content),
                "avg_chunk_size": np.mean([c['length'] for c in db.chunks])
            })
        return stats

    def save_all(self, output_dir):
        saved = {}
        for db in self.databases:
            name = db.novel_name.replace(" ", "_")
            idx_p = f"{output_dir}/{name}.index"
            meta_p = f"{output_dir}/{name}_meta.json"
            faiss.write_index(db.index, idx_p)
            with open(meta_p, 'w') as f:
                json.dump(db.chunks, f)
            saved[db.novel_name] = {"index_path": idx_p, "metadata_path": meta_p, "chunks_path": meta_p}
        return saved

    def query_both(self, query, top_k=3):
        return {db.novel_name: db.query(query, top_k) for db in self.databases}

    def test_both(self, queries):
        for q in queries:
            print(f"\n🔍 Query: {q}")
            res = self.query_both(q, top_k=1)
            for n, p in res.items():
                print(f"  [{n}]: {p[0]['text'][:80]}...")


nov_1= "The Count of Monte Cristo"
nov_2= "In Search of the Castaways"

nov_1_path= "/Users/sambit/Desktop/Hackathon IIT KGP/The-Count-of-Monte-Cristo-CLEAN.txt"
nov_2_path= "/Users/sambit/Desktop/Hackathon IIT KGP/In-search-of-the-castaways-CLEAN.txt"

def main():
    print ("Dual Vector Database creation")
    config= ChunkConfig(target_chunk_size= 1000, min_chunk_size=300, overlap_chars=150)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OUTPUT_DIR = "."
    print ("Initializeing the vector database")
    manager= DualVectorDatabaseManager(novel1_name=nov_1,
        novel1_path=nov_1_path,
        novel2_name=nov_2,
        novel2_path=nov_2_path,
        embedding_model=EMBEDDING_MODEL,
        chunk_config=config)
    print ("Building the vector databases")
    stats= manager.build_all()
    print ("The databses are created")
    saved_paths= manager.save_all(OUTPUT_DIR)
    print ("The databses are saved")
    for novel_name,paths in saved_paths.items():
        print (novel_name)
        print(f"  Index: {paths['index_path']}")
        print(f"  Metadata: {paths['metadata_path']}")
        print(f"  Chunks: {paths['chunks_path']}")

    print ("Testing vector databases")
    test_queries = [
        "The character was imprisoned wrongly",
        "He planned revenge against his enemies",
        "A wealthy man with power and influence"
    ]
    manager.test_both(test_queries)

    print ("The chunk quality")
    for db in manager.databases:
        print (novel_name)
        mid_sentence= 0
        for c in db.chunks:
            if not re.search(r"[.!?]$", c["text"].strip()):
                mid_sentence+=1
        sizes= [c["length"] for c in db.chunks]
        sentences = [c["num_sentences"] for c in db.chunks]
        paragraphs = [c["num_paragraphs"] for c in db.chunks]


        print ("Mid Sentence Chunks: ",mid_sentence)
        print ("Chunk size range: ", min(sizes),"-",max(sizes))
        print("  Avg sentences per chunk:", np.mean(sentences))
        print("  Avg paragraphs per chunk:", np.mean(paragraphs))
        config_data = {
        "novels": [
            {
                "name": nov_1,
                "path": nov_1_path,
                "index": saved_paths[nov_1]["index_path"],
                "metadata": saved_paths[nov_1]["metadata_path"],
                "chunks": saved_paths[nov_1]["chunks_path"],
                "num_chunks": stats[0]["num_chunks"],
                "total_chars": stats[0]["total_chars"]
            },
            {
                "name": nov_2,
                "path": nov_2_path,
                "index": saved_paths[nov_2]["index_path"],
                "metadata": saved_paths[nov_2]["metadata_path"],
                "chunks": saved_paths[nov_2]["chunks_path"],
                "num_chunks": stats[1]["num_chunks"],
                "total_chars": stats[1]["total_chars"]
            }
        ],
        "embedding_model": EMBEDDING_MODEL,
        "chunk_config": {
            "target_chunk_size": config.target_chunk_size,
            "min_chunk_size": config.min_chunk_size,
            "overlap_chars": config.overlap_chars
        }
    }
    config_path = f"{OUTPUT_DIR}/vector_db_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print("✓ Configuration saved:", config_path)

    print("\n" + "=" * 70)
    print("VECTOR DATABASE CREATION COMPLETE")
    print("=" * 70)

    print(f"\nDatabase 1: {nov_1}")
    print("  Chunks:", stats[0]["num_chunks"])
    print("  Avg chunk size:", int(stats[0]["avg_chunk_size"]))

    print(f"\nDatabase 2: {nov_2}")
    print("  Chunks:", stats[1]["num_chunks"])
    print("  Avg chunk size:", int(stats[1]["avg_chunk_size"]))

    print("\nTotal chunks:", stats[0]["num_chunks"] + stats[1]["num_chunks"])
    print("\nReady for RAG Pipeline Integration")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
