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

from src.core.build_scene_index import build_scene_index 

# TODO: 이 부분은 나중에 백엔드 DB에서 직접 데이터를 가져오는 함수로 대체해야 함.
MOCK_STORY_NODE_FILE="data/mock/sample_story_node.json" #Dict 형식
MOCK_SCENE_LIST_FILE="data/mock/sample_story_node.json" #List[Dict] 형식
MOCK_CHOICE_LIST_FILE ="data/mock/sample_story_node.json"#List[Dict] 형식


def _load_json_data(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"데이터 파일 {p}를 찾을 수 없습니다.")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    
    try:
        story_data = _load_json_data(MOCK_STORY_NODE_FILE)
        scene_data = _load_json_data(MOCK_SCENE_LIST_FILE)
        choice_data = _load_json_data(MOCK_CHOICE_LIST_FILE)
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("인덱스 구축에 실패했습니다.")
        return
    print("데이터 로드 완료.")

   
    build_scene_index(
        story_node_data=story_data,
        scene_node_list=scene_data,
        choice_relationship_list=choice_data
    )
    
    print("--- scene RAG 인덱스 구축 완료 ---")

if __name__ == "__main__":
    main()