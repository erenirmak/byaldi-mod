from byaldi import RAGMultiModalModel
from byaldi.colpali import ColPaliModel
import torch

import fitz  # PyMuPDF
from PIL import Image

from pathlib import Path
import os
import tempfile

from typing import Dict, List, Optional, Union

class PDFPageList:
    """
    A class that enables indexed rendering of PDF pages as images.
    """
    def __init__(self, images):
        """
        Initializes the page list with pre-processed images.

        :param images: A list of tuples (pdf_file, page_number, pixmap_data).
        """
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Render and return the image of the requested page when indexed.
        """
        if index < 0 or index >= len(self.images):
            raise IndexError("Page index out of range.")

        pdf_file, page_num, pixmap = self.images[index]

        # Convert Pixmap to PIL Image
        mode = "RGBA" if pixmap.alpha else "RGB"
        img = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)

        # Display image
        # img.show()
        return img

class PDFConverter:
    """
    A class that processes PDF files and returns an indexable list of pages.
    """
    def __init__(self, path):
        self.path = path
        self.pdf_files = self._gather_pdfs(path)

    def _gather_pdfs(self, path):
        """
        Gather all PDF files from the provided path.
        """
        str_path= str(path)
        if os.path.isfile(str_path) and str_path.lower().endswith('.pdf'):
            return [str_path]
        elif os.path.isdir(str_path):
            return [os.path.join(str_path, f) for f in os.listdir(str_path) if f.lower().endswith('.pdf')]
        else:
            raise ValueError("The provided path is neither a PDF file nor a directory containing PDF files.")

    def convert_to_image(self, zoom=1.0, output_file: any = None):
        """
        Processes all pages from the PDF(s) and returns an indexable image list.

        If output_file is provided, each page is saved to disk. If output_file is a generator,
        the next filename will be obtained for each page. Otherwise, all pages are returned as in-memory
        pixmap data wrapped in a PDFPageList.

        :param zoom: The zoom factor for rendering (default is 1.0 for 72 DPI).
        :param output_file: A filename or generator that yields filenames. If provided, pages will be saved to disk.
        :return: Either an instance of PDFPageList (if output_file is None) or a list of output file paths.
        """
        images = []

        for pdf_file in self.pdf_files:
            doc = fitz.open(pdf_file)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)  # Generate pixmap

                if output_file is not None:
                    # If output_file is a generator, get the next filename.
                    if hasattr(output_file, "__next__"):
                        filename = next(output_file)
                    else:
                        filename = output_file
                    # You may change the extension if needed. Here, we assume PNG.
                    if not filename.lower().endswith(".png"):
                        filename += ".png"
                    pix.save(filename)
                    images.append(filename)
                else:
                    images.append((pdf_file, page_num, pix))
            doc.close()

        if output_file is not None:
            return images
        else:
            return PDFPageList(images)

#### Need further modifications to the ColPaliModel and RAGMultiModalModel classes to use the above classes. ####
class ColPaliModel2(ColPaliModel):
    def __init__(self, *args, **kwargs):
        super(ColPaliModel2, self).__init__(*args, **kwargs)

    def _process_and_add_to_index(
        self,
        item: Union[Path, Image.Image],
        store_collection_with_index: bool,
        doc_id: Union[str, int],
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        """Process documents and add images to the index."""
        if isinstance(item, Path):
            if item.suffix.lower() == ".pdf":
                with tempfile.TemporaryDirectory() as path:
                    pdf_converter = PDFConverter(item)
                    images = pdf_converter.convert_to_image(
                        # You can add additional parameters like output_file if needed.
                    )
                    # images might be a PDFPageList or list of file paths.
                    if isinstance(images, PDFPageList):
                        for i in range(len(images)):
                            # images[i] returns a PIL Image.
                            image = images[i]
                            self._add_to_index(
                                image,
                                store_collection_with_index,
                                doc_id,
                                page_id=i + 1,
                                metadata=metadata,
                            )
                    else:
                        for i, item_obj in enumerate(images):
                            if isinstance(item_obj, str):
                                image = Image.open(item_obj)
                            else:
                                image = item_obj
                            self._add_to_index(
                                image,
                                store_collection_with_index,
                                doc_id,
                                page_id=i + 1,
                                metadata=metadata,
                            )
            elif item.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image = Image.open(item)
                self._add_to_index(
                    image, store_collection_with_index, doc_id, metadata=metadata
                )
            else:
                raise ValueError(f"Unsupported input type: {item.suffix}")
        elif isinstance(item, Image.Image):
            self._add_to_index(
                item, store_collection_with_index, doc_id, metadata=metadata
            )
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")

class RAGMultiModalModel2(RAGMultiModalModel):
    model: Optional[ColPaliModel2] = None

    def __init__(self, *args, **kwargs):
        super(RAGMultiModalModel2, self).__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: str = "./.byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        """Load a ColPali model from a pre-trained checkpoint.

        Parameters:
            pretrained_model_name_or_path (str): Local path or huggingface model name.
            device (str): The device to load the model on. Default is "cuda".

        Returns:
            cls (RAGMultiModalModel): The current instance of RAGMultiModalModel, with the model initialised.
        """
        instance = cls()
        instance.model = ColPaliModel2.from_pretrained(
            pretrained_model_name_or_path,
            index_root=index_root,
            device=device,
            verbose=verbose,
        )
        return instance

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        index_root: str = "./.byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        """Load an Index and the associated ColPali model from an existing document index.

        Parameters:
            index_path (Union[str, Path]): Path to the index.
            device (str): The device to load the model on. Default is "cuda".

        Returns:
            cls (RAGMultiModalModel): The current instance of RAGMultiModalModel, with the model and index initialised.
        """
        instance = cls()
        index_path = Path(index_path)
        instance.model = ColPaliModel2.from_index(
            index_path, index_root=index_root, device=device, verbose=verbose
        )

        return instance