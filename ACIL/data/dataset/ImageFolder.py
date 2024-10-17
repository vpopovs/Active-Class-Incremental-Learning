import logging
import os

from torchvision.datasets import ImageFolder as TorchImageFolder
from torchvision.datasets.folder import default_loader

from ACIL.utils.cache import cache_file

logging.getLogger("PIL").setLevel(logging.WARNING)


class ImageFolder(TorchImageFolder):
    """
    Torch's ignores the hierarchy of the folders, so this class is created to keep the hierarchy:
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    Instead, this class does:
        root/dog/xxx.png
        root/dog/xxy.png
        root/[...]/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/[...]/dog/asd932_.png
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ImageFolder object. Uses cache to store the classes.
        """
        if "loader" not in kwargs:
            kwargs["loader"] = default_loader
        self.cache = os.path.join(kwargs["root"], ".cache", "classes.pkl") if not hasattr(self, "cache") else self.cache
        super().__init__(*args, **kwargs)

    @cache_file
    def find_classes(self, root_directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Overwrites the original method to keep the hierarchy of the folders.

        Args:
            root_directory (str): Root directory of the dataset.
        """
        classes = []

        for dirpath, _, filenames in os.walk(root_directory):
            if len(filenames) == 0:
                continue
            classes.append(os.path.relpath(dirpath, root_directory))

        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class PartialImageFolder(ImageFolder):
    """Variant of ImageFolder that only loads the specified classes."""

    def __init__(self, families: list, *args, **kwargs):
        """Initializes the PartialImageFolder object. Uses cache to store the classes."""
        self.families = families
        self.cache = os.path.join(kwargs["root"], ".cache", f"classes_{'_'.join(sorted(self.families))}.pkl")
        super().__init__(*args, **kwargs)

    @cache_file
    def find_classes(self, root_directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Overwrites the original method to keep the hierarchy of the folders.

        Args:
            root_directory (str): Root directory of the dataset.
        """
        classes = []
        for family in self.families:
            for dirpath, _, filenames in os.walk(os.path.join(root_directory, family)):
                if len(filenames) == 0:
                    continue
                classes.append(os.path.relpath(dirpath, root_directory))

        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) > 0, f"Classes [{', '.join(self.families)}] not found in {root_directory}"
        return classes, class_to_idx
