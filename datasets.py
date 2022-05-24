import os
import torch
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.decomposition import PCA


TOL = 1e-5


class HyperDataset:

    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        dataset,
        train=True,
        spatial=True,
        num=1,
        neighbor=5,
        shuffle=False,
        ind=False,
    ):
        self.root = root
        self.dataset = dataset
        if spatial:
            self.data, self.label = mat_loader_spat(root, dataset, train, num=num, neighbor=neighbor, shuffle=shuffle, ind=ind)
        else:
            self.data, self.label = mat_loader_spec(root, dataset, train, shuffle=shuffle)

        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)
        self.data = self.data.unsqueeze(1)

        #print(self.data.shape)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


def mat_loader_spat(root, dataset, train, num=1, neighbor=5, shuffle=False, ind=False):

    """Read hyperspectral image data for pytorch"""

    if dataset == "houston2013":
        if train:
            data_path = os.path.join(root, "Houston2013", "Houston2013.mat")
            label_path = os.path.join(root, "Houston2013", "TRLabel.mat")
            data, label = sio.loadmat(data_path)["Houston"], sio.loadmat(label_path)["TRLabel"]
        else:
            data_path = os.path.join(root, "Houston2013", "Houston2013.mat")
            label_path = os.path.join(root, "Houston2013", "TSLabel.mat")
            data, label = sio.loadmat(data_path)["Houston"], sio.loadmat(label_path)["TSLabel"]

    elif dataset == "houston2018":
        if train:
            data_path = os.path.join(root, "Houston2018", "Houston2018.mat")
            label_path = os.path.join(root, "Houston2018", "TRLabel.mat")
            data, label = (
                sio.loadmat(data_path)["Houston"],
                sio.loadmat(label_path)["TRLabel"],
            )
        else:
            data_path = os.path.join(root, "Houston2018", "Houston2018.mat")
            label_path = os.path.join(root, "Houston2018", "TSLabel")
            data, label = (
                sio.loadmat(data_path)["Houston"],
                sio.loadmat(label_path)["TSLabel"],
            )

    elif dataset == "indian":
        
        if train:
            data_path = os.path.join(root, "Indian Pines", "Indian_pines_corrected.mat")
            label_path = os.path.join(root, "Indian Pines", str(num), "TRLabel.mat")
            print(label_path)
            data, label = (
                sio.loadmat(data_path)["indian_pines_corrected"],
                sio.loadmat(label_path)["TRLabel"],
            )
        else:
            data_path = os.path.join(root, "Indian Pines", "Indian_pines_corrected.mat")
            label_path = os.path.join(root, "Indian Pines", str(num), "TSLabel.mat")
            data, label = (
                sio.loadmat(data_path)["indian_pines_corrected"],
                sio.loadmat(label_path)["TSLabel"],
            )

    elif dataset == "pavia":

        if train:
            data_path = os.path.join(root, "Pavia University", "PaviaU.mat")
            label_path = os.path.join(root, "Pavia University", str(num), "TRLabel.mat")
            data, label = (
                sio.loadmat(data_path)["paviaU"],
                sio.loadmat(label_path)["TRLabel"],
            )
        else:
            data_path = os.path.join(root, "Pavia University", "PaviaU.mat")
            label_path = os.path.join(root, "Pavia University", str(num), "TSLabel.mat")
            data, label = (
                sio.loadmat(data_path)["paviaU"],
                sio.loadmat(label_path)["TSLabel"],
            )

    elif dataset == "salinas":
        if train:
            data_path = os.path.join(root, "Salinas", "Salinas_corrected.mat")
            label_path = os.path.join(root, "Salinas", str(num), "TRLabel.mat")
            data, label = (
                sio.loadmat(data_path)["salinas_corrected"],
                sio.loadmat(label_path)["TRLabel"],
            )
        else:
            data_path = os.path.join(root, "Salinas", "Salinas_corrected.mat")
            label_path = os.path.join(root, "Salinas", str(num), "TSLabel.mat")
            data, label = (
                sio.loadmat(data_path)["salinas_corrected"],
                sio.loadmat(label_path)["TSLabel"],
            )

    data, label = np.array(data).astype(float) / 1.0, np.array(label).astype(int)

    # 主成分分析
    if ind==True:
        dims=data.shape
        
        data = data.reshape(-1,dims[2]).T
        #print(type(data))
        pca = PCA(n_components=30)

        pca.fit(data)
        data=pca.components_
        data=data.T.reshape(dims[0],dims[1],30)
    #print(data.shape)

    #print(data)

    #print(np.min(data))

    #print(np.max(data))
    #print(data)

    data = (
        (data - np.min(data)) / (np.max(data) - np.min(data)) * 1000
    )
    #print('data:', data.shape)
    #print(np.min(data))
    
    #data = minmax_scale(data)

    #print('data:', data.shape)
    # print('label:', label.shape)

    ix = np.where(label > TOL)[0]
    iy = np.where(label > TOL)[1]

    
    #if(train==False):
    #    ix = ix[:-1]
    #    iy = iy[:-1]
    #print('ix:', ix.shape)
    #print('iy:', iy.shape)

    data_label = label[tuple([ix, iy])] - 1.0

    data_vec = np.zeros((len(ix), np.size(data, 2), neighbor, neighbor))

    data_extent = np.zeros((data.shape[0]+neighbor-1, data.shape[1]+neighbor-1, data.shape[2]))
    #print(data_extent.shape)
    data_extent[np.floor(neighbor / 2).astype(int):np.floor(neighbor / 2).astype(int)+data.shape[0], np.floor(neighbor / 2).astype(int):np.floor(neighbor / 2).astype(int)+data.shape[1], :] = data[:,:,:]

    #print(np.floor(neighbor / 2).astype(int),np.floor(neighbor / 2).astype(int)+data.shape[0])
    #print(np.floor(neighbor / 2).astype(int),np.floor(neighbor / 2).astype(int)+data.shape[1])

    for i in range(np.floor(neighbor / 2).astype(int)):
        data_extent[:,i,:]=data_extent[:,np.floor(neighbor / 2).astype(int),:]
        data_extent[i,:,:]=data_extent[np.floor(neighbor / 2).astype(int),:,:]
        data_extent[:,data.shape[1]+i+np.floor(neighbor / 2).astype(int),:]=data_extent[:,data.shape[1]+np.floor(neighbor / 2).astype(int)-1,:]
        data_extent[data.shape[0]+i+np.floor(neighbor / 2).astype(int),:,:]=data_extent[data.shape[2]+np.floor(neighbor / 2).astype(int)-1,:,:]

    #print(data_extent[349,5,5])
    #print(data_extent[350,5,5])
    #print(data_extent[351,5,5])
    #print(data_extent[352,5,5])

    for i in range(neighbor):
        for j in range(neighbor):
            data_vec[:, :, i, j] = data_extent[
                tuple(
                    [
                        ix + i,
                        iy + j,
                    ]
                )
            ]

    if shuffle:
        index = np.arange(len(data_label))
        np.random.shuffle(index)

        data_vec = data_vec[index, ...]
        data_label = data_label[index]
    else:
        pass
    #print(data_label[0:200])
    return data_vec, data_label.astype("long")


def mat_loader_spec(root, dataset, train, shuffle=False):

    """Read hyperspectral image data for pytorch"""

    if dataset == "houston2013":
        data_path = os.path.join(root, "Houston2013", "Houston.mat")
        label_path = os.path.join(root, "Houston2013", "Houston_gt.mat")
        data, label = sio.loadmat(data_path)["Houston"], sio.loadmat(label_path)["gt"]

    elif dataset == "houston2018":
        data_path = os.path.join(root, "Houston2018", "Houston2018.mat")
        label_path = os.path.join(root, "Houston2018", "Houston2018_gt.mat")
        data, label = (
            sio.loadmat(data_path)["Houston2018"],
            sio.loadmat(label_path)["Houston2018_gt"],
        )

    elif dataset == "indian":
        data_path = os.path.join(root, "Indian_Pines", "Indian_pines_corrected.mat")
        label_path = os.path.join(root, "Indian_Pines", "Indian_pines_gt.mat")
        data, label = (
            sio.loadmat(data_path)["indian_pines_corrected"],
            sio.loadmat(label_path)["indian_pines_gt"],
        )

    elif dataset == "pavia":
        data_path = os.path.join(root, "university pavia", "PaviaU.mat")
        label_path = os.path.join(root, "university pavia", "PaviaU_gt.mat")
        data, label = (
            sio.loadmat(data_path)["paviaU"],
            sio.loadmat(label_path)["paviaU_gt"],
        )

    elif dataset == "salinas":
        data_path = os.path.join(root, "salina", "Salinas_corrected.mat")
        label_path = os.path.join(root, "salina", "Salinas_gt.mat")
        data, label = (
            sio.loadmat(data_path)["salinas_corrected"],
            sio.loadmat(label_path)["salinas_gt"],
        )

    else:
        pass

    data, label = np.array(data) / 1.0, np.array(label).astype(int)

    # print('data:', data.shape)
    # print('label:', label.shape)

    ix = np.where(label > TOL)[0]
    iy = np.where(label > TOL)[1]

    data_vec = data[tuple([ix, iy])]

    data_vec = (data_vec - np.min(data_vec)) / (np.max(data_vec) - np.min(data_vec))

    data_label = label[tuple([ix, iy])] - 1.0

    if shuffle:
        index = np.arange(len(data_label))
        np.random.shuffle(index)

        data_vec = data_vec[index, ...]
        data_label = data_label[index]
    else:
        pass
    print(data_label[0:200])

    return data_vec, data_label.astype(int)


def data_info(dataset):
    if dataset == "pavia":
        input_dim = 103
        num_classes = 9
    elif dataset == "indian":
        input_dim = 200
        num_classes = 16
    elif dataset == "salinas":
        input_dim = 204
        num_classes = 16
    elif dataset == "houston2013":
        input_dim = 144
        num_classes = 15
    elif dataset == "houston2018":
        input_dim = 48
        num_classes = 20
    else:
        raise LookupError("Wrong dataset")
    return input_dim, num_classes


if __name__ == "__main__":

    root = "/home/hyper/hyperspectral_dataset"
    dataset = "houston2018"

    data_vec, data_label = mat_loader_spat(root, dataset, neighbor=5)

    print(data_vec.shape)
    print(data_label.shape)

    print(len(data_vec[1]))
    print(len(data_label))
