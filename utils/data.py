import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import sys
import os
class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iTinyImageNet200(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(64),
    ]
    common_trsf = [
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = "/data/open-datasets/tiny-imagenet-200/train"
        test_dir = "/data/open-datasets/tiny-imagenet-200/val"
        train_dset = datasets.ImageFolder(train_dir)

        tmp_train_im_ids = [
        'n04311004', 'n02099601', 'n02814860', 'n03983396', 'n02843684',
        'n02281406', 'n03085013', 'n04008634', 'n04074963', 'n03854065',
        'n03838899', 'n04501370', 'n04366367', 'n01917289', 'n01784675',
        'n03355925', 'n03400231', 'n02666196', 'n02988304', 'n03649909',
        'n03992509', 'n03126707', 'n03617480', 'n02480495', 'n04456115',
        'n03976657', 'n07768694', 'n01629819', 'n07615774', 'n03160309',
        'n02268443', 'n03179701', 'n03930313', 'n02395406', 'n04596742',
        'n02403003', 'n09246464', 'n07695742', 'n07920052', 'n02423022',
        'n03100240', 'n03763968', 'n02795169', 'n02999410', 'n02129165',
        'n01910747', 'n01855672', 'n02236044', 'n04285008', 'n03970156',
        'n03804744', 'n02415577', 'n03902125', 'n01774750', 'n03977966',
        'n02481823', 'n01698640', 'n03891332', 'n04328186', 'n04133789',
        'n02509815', 'n02769748', 'n03937543', 'n02123394', 'n07583066',
        'n04540053', 'n04099969', 'n03393912', 'n02125311', 'n02504458',
        'n02437312', 'n09428293', 'n03637318', 'n02909870', 'n01770393',
        'n02814533', 'n01742172', 'n02815834', 'n02231487', 'n03089624',
        'n02190166', 'n04417672', 'n02808440', 'n02977058', 'n02802426',
        'n04118538', 'n02124075', 'n02699494', 'n02321529', 'n02113799',
        'n04023962', 'n04254777', 'n07720875', 'n04371430', 'n02841315',
        'n04487081', 'n01950731', 'n06596364', 'n02165456', 'n02883205',
        'n01983481', 'n04507155', 'n07873807', 'n07715103', 'n02094433',
        'n04532670', 'n02132136', 'n02963159', 'n03201208', 'n01644900',
        'n02669723', 'n04465501', 'n02892201', 'n07579787', 'n02823428',
        'n02837789', 'n02486410', 'n02788148', 'n03837869', 'n02948072',
        'n04275548', 'n04149813', 'n02085620', 'n07753592', 'n02791270',
        'n01641577', 'n03796401', 'n02730930', 'n02074367', 'n04532106',
        'n04070727', 'n01443537', 'n03255030', 'n04251144', 'n02364673',
        'n03404251', 'n03599486', 'n03706229', 'n04398044', 'n02099712',
        'n04562935', 'n09256479', 'n04399382', 'n07734744', 'n03447447',
        'n01768244', 'n02917067', 'n03584254', 'n03980874', 'n03814639',
        'n07614500', 'n04486054', 'n03014705', 'n02058221', 'n04265275',
        'n02002724', 'n03733131', 'n07875152', 'n12267677', 'n03388043',
        'n02927161', 'n03662601', 'n04259630', 'n04356056', 'n02206856',
        'n01945685', 'n04560804', 'n07747607', 'n02279972', 'n03042490',
        'n09332890', 'n04179913', 'n03026506', 'n07871810', 'n03770439',
        'n02410509', 'n03250847', 'n03424325', 'n07711569', 'n04146614',
        'n03444034', 'n02123045', 'n04067472', 'n02793495', 'n01774384',
        'n03544143', 'n01882714', 'n02226429', 'n02906734', 'n04597913',
        'n09193705', 'n01984695', 'n02056570', 'n03670208', 'n07749582',
        'n01944390', 'n04376876', 'n02233338', 'n02106662', 'n02950826']

        tmp_dct = dict(zip(tmp_train_im_ids, range(len(tmp_train_im_ids))))
        train_dset.class_to_idx = tmp_dct
        train_dset.samples = train_dset.make_dataset(train_dset.root, train_dset.class_to_idx, train_dset.extensions)
        train_dset.imgs = train_dset.samples

        # print(train_dset.class_to_idx)
        # input()

        train_images = []
        train_labels = []
        for item in train_dset.imgs:
            train_images.append(item[0])
            train_labels.append(item[1])
            # print(item[0], item[1])
            # input()
        self.train_data, self.train_targets = np.array(train_images), np.array(train_labels)

        # print(train_dset.class_to_idx)
        # input()

        test_images = []
        test_labels = []
        _, class_to_idx = find_classes(train_dir)

        class_to_idx = tmp_dct
        
        imgs_path = os.path.join(test_dir, 'images')
        imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())
        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        # print(cls_map)
        # input()
        for imgname in sorted(os.listdir(imgs_path)):
            if cls_map[imgname] in sorted(class_to_idx.keys()):
                path = os.path.join(imgs_path, imgname)
                test_images.append(path)
                test_labels.append(class_to_idx[cls_map[imgname]])
        self.test_data, self.test_targets = np.array(test_images), np.array(test_labels)


def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx














import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

class myCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class myCIFAR100(myCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
