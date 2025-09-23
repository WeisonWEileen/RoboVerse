import torch


def double_backward_cudnn_rnn():
    # 1. 构造一个单层 LSTM（CuDNN 后端默认启用）
    input_size, hidden_size = 10, 8
    lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1).cuda()

    # 2. 随机输入序列，要求梯度
    seq_len, batch = 5, 3
    x = torch.randn(seq_len, batch, input_size, device="cuda", requires_grad=True)

    # 3. 正常前向 → 一阶梯度（保留计算图）
    output, _ = lstm(x)
    loss_first = output.sum()

    # 一阶：∂loss_first / ∂x  ，create_graph=True ⇒ 保留图
    grad_x = torch.autograd.grad(loss_first, x, create_graph=True)[0]

    # 4. 构造二阶标量，再反向 ⇒ 触发 double-backward
    # loss_second = grad_x.pow(2).mean()

    # 5. 这里会抛错
    grad_x.backward()


if __name__ == "__main__":
    try:
        double_backward_cudnn_rnn()
    except NotImplementedError as e:
        print("捕获到 PyTorch 报错：")
        print(e)
