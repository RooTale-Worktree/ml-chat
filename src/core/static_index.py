import json
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbeddings
from pathlib import Path
from typing import List

# RAG가 사용할 임베딩 모델 
EMBED_MODEL_NAME = "jhgan/ko-sbert-nli"


def build_static_index(json_path: str, story_title: str):
   
   save_dir = Path("data/story_indexes") / story_title

    # 1. 표준 JSON 파일 읽기
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            story_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: JSON 파일을 찾을 수 없습니다. 경로: {json_path}")
        return
    except json.JSONDecodeError:
        print(f"오류: JSON 파일 형식이 잘못되었습니다. 경로: {json_path}")
        return

    # 2. JSON을 LlamaIndex의 'TextNode' 리스트로 변환
    nodes: List[TextNode] = []
    for item in story_data:
        # 메타데이터 구조를 TextNode로 변환
        node = TextNode(
            text=item.get("content"),
            id_=item.get("chunk_id"),  # 고유 ID로 chunk_id 사용
            metadata={
                "source_document": item.get("source_document"),
                "chunk_type": item.get("chunk_type"),
                # meta 딕셔너리 안의 모든 키-값을 상위 metadata로 병합
                **(item.get("meta", {})) 
            }
        )
        nodes.append(node)
    
    if not nodes:
        print("경고: JSON 파일에서 변환할 노드를 찾지 못했습니다.")
        return


    # 3. LlamaIndex가 사용할 임베딩 모델 설정
    Settings.embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # 4. LlamaIndex로 인덱스 생성
    print("임베딩 및 인덱싱 시작...")
    index = VectorStoreIndex(nodes, show_progress=True)
    print("인덱싱 완료.")

    # 5. 인덱스를 파일로 저장 (persist)
    Path(save_dir).mkdir(parents=True, exist_ok=True) # 저장 폴더가 없으면 생성
    index.storage_context.persist(persist_dir=save_dir)
    
    print(f"'{save_dir}'에 Static RAG 인덱스를 성공적으로 저장했습니다.")