import unittest
from transformer_library import *


class BasicTest(unittest.TestCase):
    def test_example1(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
        target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

        source_pad_index = 0
        target_pad_index = 0
        source_vocab_size = 10
        target_vocab_size = 10

        model = Transformer(source_vocab_size, target_vocab_size, source_pad_index, target_pad_index).to(device)
        out = model(x, target[:, :-1])
        assert out.shape == torch.Size([2, 7, 10])


if __name__ == '__main__':
    unittest.main()
