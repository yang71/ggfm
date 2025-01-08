import time
import random
from sklearn.model_selection import train_test_split

from warnings import filterwarnings
filterwarnings("ignore")


def construct_link_and_node(graph, data_dir):
    r"""

    Construct link.dat and node.dat.

    Parameters
    ----------
    graph: class:`ggfm.data.Graph`
        Target graph.
    
    data_dir: str
        Data directory for saving link.dat and node.dat, which are saved as data_dir/link.dat and data_dir/node.dat.
    """

    type_begin_ids, graph_node_type = get_type_id(graph)
    
    graph_node_name = {}
    for i in range(len(graph_node_type)):
        attr = "name"
        if graph_node_type[i] == "paper": attr = "title"
        graph_node_name[graph_node_type[i]] = graph.node_feature[graph_node_type[i]][attr].tolist()

    # node.dat
    node_type2id = {}
    type_num = 0
    for i in range(len(graph_node_type)):
        node_type2id[graph_node_type[i]] = type_num
        type_num += 1
    
    with open(data_dir + "node.dat", "w") as file:
        for i in range(len(graph_node_type)):
            for j in range(len(graph_node_name[graph_node_type[i]])):
                node_id = j + type_begin_ids[i]
                node_name = graph_node_name[graph_node_type[i]][j]
                node_type = node_type2id[graph_node_type[i]]
                file.write(f"{node_id}\t{node_name}\t{node_type}\n")
    
    print("node.dat has been saved.")
        
    
    # link.dat
    # start_id  end_id  edge_type
    edges = graph.edge_list
    relations = graph.get_meta_graph()
    edge_types = {}
    num_edge = 0
    for r in relations:
        edge_types[r[2]] = num_edge
        num_edge += 1
    
    with open(data_dir + "link.dat", "w") as file:
        for target_type in edges:
            for source_type in edges[target_type]:
                for relation_type in edges[target_type][source_type]:
                    for target_id in edges[target_type][source_type][relation_type]:
                        for source_id in edges[target_type][source_type][relation_type][target_id]:
                            src = source_id + type_begin_ids[node_type2id[source_type]]
                            dst = target_id + type_begin_ids[node_type2id[target_type]]
                            edge_type = edge_types[relation_type]
                            file.write(f"{src}\t{dst}\t{edge_type}\n")

    print("link.dat has been saved.")


def get_type_id(graph):
    r"""

    Statistically analyze the type_degin_ids and graph_node_type of graphs.

    Parameters
    ----------
    graph: class:`ggfm.data.Graph`
        Target graph.
    
    Returns
    -------
    type_begin_ids: list
        The begin index of each node type, type_begin_ids are consistent with graph_node_type.
    graph_node_type: list
        Graph node types.
    """

    graph_node_name = {}
    graph_node_type = graph.get_types()
    for i in range(len(graph_node_type)):
        attr = "name"
        if graph_node_type[i] == "paper": attr = "title"
        graph_node_name[graph_node_type[i]] = graph.node_feature[graph_node_type[i]][attr].tolist()

    type_begin_ids = [0, ]
    for i in range(1, len(graph_node_type)):
        type_begin_ids.append(type_begin_ids[i-1]+len(graph_node_name[graph_node_type[i-1]]))
    
    return type_begin_ids, graph_node_type



def random_walk_based_corpus_construction(data_dir, relations, alpha=0.05, path_length=1000000, path_num=450000):
    r"""

    Construct link.dat and node.dat.

    Parameters
    ----------
    data_dir: str
        Data directory for loading link.dat and node.dat, also for saving output.txt, rw_train_corpus.txt and rw_valid_corpus.txt.
    
    relations: list
        Relations for all edge types.
    
    alpha: str, optional
        Each path will terminate sampling with a probability of alpha.
        (default: :obj:`0.05`)
    
    path_length: int, optional
        Sampling length of each path.
        (default: :obj:`1000000`)
    
    path_num: int, optional
        Number of sampled paths.
        (default: :obj:`450000`)

    """

    op1, op2, op3 = [], [], []

    with open(data_dir + 'node.dat','r') as file:
        for line in file:
            node_id, node_name, node_type = line.split('\t')
            op1.append(int(node_id))
            op2.append(node_name)
            op3.append(int(node_type))  
        

    G=[[] for i in range(len(op3))]

    with open(data_dir + 'link.dat', 'r') as file:
        for line in file:
            src, dst, edge_type = line.split('\t')
            G[int(src)].append([int(dst), int(edge_type)])

    line_idx = op1
    rand = random.Random()
    patient_patient_path = []

    dic = {}
    start_time = time.time()
    for line in range(path_num):
        if line % 10000 == 0:
            current_time = time.time()
            dual_time = current_time - start_time
            print(f"having sampling {line} lines and spent {dual_time}...")
        temp_path = []
        start_path = rand.choice(line_idx)
        temp_path.append([start_path,-1])
        dic[start_path] = 1
        for i in range(path_length):
            cur = temp_path[-1][0]
            if (len(G[cur]) > 0):
                if rand.random() >= alpha:
                    cur_path = rand.choice(G[cur])
                    temp_path.append(cur_path)
                    dic[cur_path[0]] = 1
                else:
                    break
            else:
                break
        if (len(temp_path) >= 2):
            patient_patient_path.append(temp_path)


    line_name = {}
    for i in range(len(relations)):
        line_name[i] = relations[i]


    with open(data_dir + 'output.txt', 'w') as f:
        for i in range(len(patient_patient_path)):
            print(op2[patient_patient_path[i][0][0]],line_name[patient_patient_path[i][1][1]],op2[patient_patient_path[i][1][0]],end='',file=f)
            for j in range(1,len(patient_patient_path[i])-2):
                print(' '+line_name[patient_patient_path[i][j+1][1]],op2[patient_patient_path[i][j+1][0]],end='',file=f)
            if(len(patient_patient_path[i])>2):
                print(' '+line_name[patient_patient_path[i][-1][1]],op2[patient_patient_path[i][-1][0]],end='',file=f)
            print("\n",end='',file=f)


    with open(data_dir + 'output.txt', 'r') as file:
        corpus = [line.rstrip("\n") for line in file.readlines()]
    
    print(f"length of corpus: {len(corpus)}")
    print("corpus[0]: ")
    print(corpus[0])

    train_text, val_text = train_test_split(corpus, test_size=0.15, random_state=42)

    with open(data_dir + 'rw_train_corpus.txt', 'w') as file:
        for paragraph in train_text:
            file.write(paragraph + "\n")
            
    with open(data_dir + 'rw_val_corpus.txt', 'w') as file:
        for paragraph in val_text:
            file.write(paragraph + "\n")

    print(f"length of train_corpus: {len(train_text)}")
    print(f"length of valid_corpus: {len(val_text)}")
    print("train_corpus and valid_corpus have been saved.")