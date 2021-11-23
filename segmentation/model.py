from segmentation.encoders import Encoder
from segmentation.decoders import Decoder
from segmentation.base import SegmentationModel
from segmentation.base import SegmentationHead, ClassificationHead


class SegModel(SegmentationModel):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        segmentation_head: SegmentationHead,
        classification_head: ClassificationHead
    ):
        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.segmentation_head = segmentation_head

        if self.classification_head is not None:
            self.classification_head = classification_head
        else:
            self.classification_head = None

        self.name = "{}{}".format(encoder.__name__, decoder.__name__)
        self.initialize()
