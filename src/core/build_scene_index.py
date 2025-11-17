import json
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbeddings
from pathlib import Path
from typing import List, Dict, Any


# RAG가 사용할 임베딩 모델
EMBED_MODEL_NAME = "jhgan/ko-sbert-nli"

BASE_INDEX_DIR = "data/story_indexes" 

def _parse_story_node(story_node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    백엔드의 (Story) 노드 1개를 파싱하여
    RAG에 필요한 핵심 정보(제목, ID, 작가)를 추출.
    """
    story_info = {
        "story_id": story_node_data.get("story_id"),
        "title": story_node_data.get("title"),
        "author": story_node_data.get("author")
    }
    
    if not story_info["title"]:
        raise ValueError("입력된 Story 노드 데이터에 'title' 필드가 없습니다.")
    return story_info

def _convert_scene_to_textnode(scene_data: Dict[str, Any], story_title: str) -> TextNode:
    """
    백엔드의 (Scene) 노드 1개를 LlamaIndex의 TextNode 1개로 변환.
    """
    metadata = {
        "source_document": story_title,
        "chunk_type": "story_node" 
    }

    
    for key, value in scene_data.items():
        if key in ["story_text", "node_id"]:
            continue
        
        if isinstance(value, (dict, list)):
            metadata[key] = json.dumps(value)
        else:
            metadata[key] = value

    return TextNode(
        text=scene_data.get("story_text", ""), 
        id_=scene_data.get("node_id"), 
        metadata=metadata
    )

def _convert_choice_to_textnode(choice_data: Dict[str, Any], story_title: str) -> TextNode:
    """
    백엔드의 [:CHOICE] 관계 1개를 LlamaIndex의 TextNode 1개로 변환
    """
    parent_node_id = choice_data.get("parent_node_id")
    next_node_id = choice_data.get("next_node_id")     

    metadata = {
        "source_document": story_title,
        "chunk_type": "choice_node",
        "parent_node_id": parent_node_id, 
        "next_node_id": next_node_id      
    }

    return TextNode(
        text=choice_data.get("choice_text", ""), 
        id_=f"choice-{parent_node_id}-{next_node_id}", 
        metadata=metadata
    )

def build_scene_index(
    story_node_data: Dict[str, Any],
    scene_node_list: List[Dict[str, Any]], 
    choice_relationship_list: List[Dict[str, Any]]
):
   
    
    story_info = _parse_story_node(story_node_data)
    story_title = story_info["title"] 

    save_dir = Path(BASE_INDEX_DIR) / story_title

    scene_nodes = [
        _convert_scene_to_textnode(scene, story_title) 
        for scene in scene_node_list
    ]
    print(f"총 {len(scene_nodes)}개의 Scene 노드를 변환했습니다.")

    choice_nodes = [
        _convert_choice_to_textnode(choice, story_title) 
        for choice in choice_relationship_list
    ]
    print(f"총 {len(choice_nodes)}개의 Choice 노드를 변환했습니다.")

    all_nodes = scene_nodes + choice_nodes
    if not all_nodes:
        print("경고: 변환할 노드가 없습니다. 인덱스를 생성하지 않습니다.")
        return

    Settings.embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    print("임베딩 및 인덱싱 시작...")
    index = VectorStoreIndex(all_nodes, show_progress=True)
    print("인덱싱 완료.")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(save_dir))
    
    print(f"'{save_dir}'에 Static RAG 인덱스를 성공적으로 저장했습니다.")

    story_info_path = save_dir / "story_info.json"
    with story_info_path.open("w", encoding="utf-8") as f:
        json.dump(story_info, f, ensure_ascii=False, indent=4)
   