from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.graphs.index_creator import GraphIndexCreator

# Ollama LLM 모델 초기화 
# 필요에 따라 모델명 변경 가능 (llama2, mistral, mixtral 등)
llm = Ollama(model="llama3.1", temperature=0)

# 방법 1: LLMGraphTransformer를 사용하여 그래프 생성
def create_graph_from_text(text):
    documents = [Document(page_content=text)]
    llm_transformer_filtered = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Person", "Country", "Organization"],
        allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
    )
    graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)
    
    graph = NetworkxEntityGraph()
    
    # 노드 추가
    for node in graph_documents_filtered[0].nodes:
        graph.add_node(node.id)
    
    # 엣지 추가
    for edge in graph_documents_filtered[0].relationships:
        graph._graph.add_edge(
            edge.source.id,
            edge.target.id,
            relation=edge.type,
        )
    
    return graph

# 방법 2: GraphIndexCreator를 사용하여 그래프 생성
def create_graph_from_file(file_path):
    try:
        with open(file_path) as f:
            text = f.read()
        
        index_creator = GraphIndexCreator(llm=llm)
        return index_creator.from_text(text)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None

# 예제 텍스트 
sample_text = """
마리 퀴리는 1867년에 태어난 폴란드 출신의 프랑스계 물리학자이자 화학자로 방사능에 대한 선구적 연구를 수행했습니다. 
그녀는 노벨상을 수상한 최초의 여성이었고, 노벨상을 두 번 수상한 최초의 사람이었으며, 두 과학 분야에서 노벨상을 수상한 유일한 사람이었습니다. 
그녀의 남편 피에르 퀴리는 그녀의 첫 노벨상을 공동 수상한 사람이었고, 그들은 역사상 최초의 결혼한 사람이 되었습니다. 부부가 노벨상을 수상하고 퀴리 가문의 5개 노벨상 유산을 시작했습니다. 
그녀는 1906년 파리 대학에서 교수가 된 최초의 여성이었습니다.
"""

# 두 가지 방법 중 하나를 선택하여 사용
# 방법 1: 텍스트에서 직접 그래프 생성
graph = create_graph_from_text(sample_text)

# 방법 2: 파일에서 그래프 생성 (필요시 주석 해제)
# file_path = "sample.txt"  # 상대 경로 사용
# graph = create_graph_from_file(file_path)

# 그래프 기반 QA 체인 생성
chain = GraphQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# 한국어 질문으로 테스트
response = chain.run("피에르 퀴리는 어떤 상을 받았나요?")
print(response)

