#!/usr/bin/env python3
"""
Stub script to build vector indexes (chat/story) for future RAG.
Currently just creates a placeholder file in data/indexes.
"""
"""
from pathlib import Path

INDEX_DIR = Path("data/indexes")


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    placeholder = INDEX_DIR / "README.txt"
    placeholder.write_text("Placeholder for vector indexes. Replace with FAISS/Chroma build.", encoding="utf-8")
    print(f"Wrote placeholder index file at {placeholder}")

if __name__ == "__main__":
    main()
"""
"""
백엔드 DB에서 데이터를 가져와 src/core/build_scene_index.py를 호출.
"""
import json
from pathlib import Path
from typing import Any, List, Dict 
from neo4j import GraphDatabase
from src.core.build_scene_index import build_scene_index 

URI = "neo4j+ssc://32adcd36.databases.neo4j.io"
AUTH = ("neo4j", "sKyJKxvWChIunry20Sk2cA-Wi-d-0oZH75LWcZz6zUg")

def get_driver():
    return GraphDatabase.driver(URI, auth=AUTH)

def run_cypher(query: str) -> List[Dict[str, Any]]:
    records = []
    with get_driver() as driver:
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                records.append(record.data()) 
    return records

def main():
    
    query_story = "MATCH (s:Story {story_id: 'main_story'}) RETURN s" 
    story_data_list = run_cypher(query_story)
    if not story_data_list:
        print("오류: Story 노드를 찾을 수 없습니다. story_id를 확인하세요.")
        return
    
    story_node = story_data_list[0]['s']
    story_data = dict(story_node)

    query_scenes = "MATCH (n:Scene) RETURN n"
    scene_data_list = run_cypher(query_scenes)
    scene_data_list = [dict(record['n']) for record in scene_records]
    print(f"Scene 노드 {len(scene_data_list)}개 추출 완료.")

    
    query_choices = """
    MATCH (sc1:Scene)-[r:CHOICE]->(sc2:Scene) 
    RETURN sc1.node_id AS parent_node_id, 
           r.choice_text AS choice_text, 
           sc2.node_id AS next_node_id
    """
    choice_data_list = run_cypher(query_choices)
    print(f"Choice 관계 {len(choice_data_list)}개 추출 완료.")

   
    build_scene_index(
        story_node_data=story_data,
        scene_node_list=scene_data_list, 
        choice_relationship_list=choice_data_list
    )
    
    print("--- scene RAG 인덱스 구축 완료 ---")

if __name__ == "__main__":
    main()