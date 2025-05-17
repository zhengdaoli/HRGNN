import numpy as np


class GraphDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_targets(self):

        if len(self.data[0].y.shape) > 0 and self.data[0].y.shape[0] > 1:
            return np.stack([d.y for d in self.data], axis=0)
        else:
            return np.array([d.y.item() for d in self.data])

    def get_data(self):
        return self.data

    def augment(self, v_outs=None, e_outs=None, g_outs=None, o_outs=None):
        """
        v_outs must have shape |G|x|V_g| x L x ? x ...
        e_outs must have shape |G|x|E_g| x L x ? x ...
        g_outs must have shape |G| x L x ? x ...
        o_outs has arbitrary shape, it is a handle for saving extra things
        where    L = |prev_outputs_to_consider|.
        The graph order in which these are saved i.e. first axis, should reflect the ones in which
        they are saved in the original dataset.
        :param v_outs:
        :param e_outs:
        :param g_outs:
        :param o_outs:
        :return:
        """
        for index in range(len(self)):
            if v_outs is not None:
                self[index].v_outs = v_outs[index]
            if e_outs is not None:
                self[index].e_outs = e_outs[index]
            if g_outs is not None:
                self[index].g_outs = g_outs[index]
            if o_outs is not None:
                self[index].o_outs = o_outs[index]


class GraphDatasetSubset(GraphDataset):
    """
    Subsets the dataset according to a list of indices.
    """

    def __init__(self, data, indices, fake_edges=None):
        self.data = data
        self.indices = indices
        self.fake_edges = fake_edges

    def __getitem__(self, index):
        cur_data = self.data[self.indices[index]]
        
        if self.fake_edges is not None:
            # print('before edge index', cur_data.edge_index)
            cur_data.edge_index = self.fake_edges[self.indices[index]].clone()
            # print('after edge index', cur_data.edge_index)
            
        return cur_data

    def get_subset(self):
        sub_data = [self.data[i] for i in self.indices]
        return sub_data
        
    def __len__(self):
        return len(self.indices)

    def get_targets(self):
        if self.data[0].y.shape[0] > 1:
            return np.stack([self.data[i].y for i in self.indices], axis=0)
        else:
            return np.array([self.data[i].y.item() for i in self.indices])
        
