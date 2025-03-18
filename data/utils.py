from typing import List, Dict, Tuple
from collections import deque
from torch.utils.data import Dataset, DataLoader


def read_triples(file_path: str):
    triples = list()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = line.strip().split()
            triples.append((h, r, t))
    return triples


def get_hr2t_and_tr2h(triples: List[Tuple]) -> Tuple[Dict, Dict]:
    hr2t, tr2h = dict(), dict()
    for h, r, t in triples:
        if (h, r) not in hr2t:
            hr2t[h, r] = set()
        hr2t[h, r].add(t)

        if (t, r) not in tr2h:
            tr2h[t, r] = set()
        tr2h[t, r].add(h)
    return hr2t, tr2h


class KGCDataset(Dataset):
    def __init__(self, data: List[Tuple]):
        super(KGCDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


class LinkGraph:
    def __init__(self, triples: List[Tuple]):
        self.graph = {}
        for h, r, t in triples:
            if h not in self.graph:
                self.graph[h] = set()
            if t not in self.graph:
                self.graph[t] = set()
            self.graph[h].add(t)
            self.graph[t].add(h)

    def get_neighbor_ents(self, ent: str, max_to_keep=10) -> List[str]:
        neighbor_ents = self.graph.get(ent, set())
        return sorted(list(neighbor_ents))[: max_to_keep]

    def get_n_hop_ents(self, ent: str, n_hop: int = 2, max_nodes: int = 100000) -> List:
        if n_hop < 0:
            return list()

        seen_eids = set()
        seen_eids.add(ent)
        queue = deque([ent])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return list()
        # return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])
        return list(seen_eids)
