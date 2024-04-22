import torch.nn.functional as F


class Loss_functions:
    def get_loss_func(self, query):
        if query == "clip_bce":
            return self.clip_bce
        elif query == "ce":
            return self.ce

    def clip_bce(self, output_dict, target_dict):
        """Binary crossentropy loss."""
        return F.binary_cross_entropy(
            output_dict["clipwise_output"], target_dict["target"]
        )

    def ce(self, output_dict: dict, target_dict: dict):
        return F.cross_entropy(output_dict["clipwise_output"], target_dict["target"])
