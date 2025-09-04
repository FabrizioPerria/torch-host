#include "NeuralNetwork.h"
#include <ATen/ops/mse_loss.h>
#include <iostream>
#include <random>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>

std::vector<std::pair<float, float>> getLine (float bias, float weight, float startX, float endX, float count)
{
    float y { 0 };
    std::vector<std::pair<float, float>> line;
    float dX = (endX - startX) / count;
    for (float x = startX; x < endX; x += dX)
    {
        y = bias + weight * x;
        line.push_back (std::make_pair (x, y));
    }
    return line;
}

void testLine()
{
    float bias = 0.0f;
    float weight = 0.0f;
    float startX = 0.0f;
    float endX = 10.0f;
    float count = 10.0f;

    auto line = getLine (bias, weight, startX, endX, count);
    assert (line.size() == count);
    assert (std::all_of (line.begin(), line.end(), [] (const auto& point) { return point.second == 0.0f; }));

    weight = 1.0f;
    line = getLine (bias, weight, startX, endX, count);
    assert (line.size() == count);
    assert (std::all_of (line.begin(), line.end(), [] (const auto& point) { return point.second == point.first; }));

    bias = 1.0f;
    line = getLine (bias, weight, startX, endX, count);
    assert (line.size() == count);
    assert (std::all_of (line.begin(), line.end(), [] (const auto& point) { return point.second == 1.0f + point.first; }));

    weight = -2.0f;
    line = getLine (bias, weight, startX, endX, count);
    assert (line.size() == count);
    assert (std::all_of (line.begin(), line.end(), [] (const auto& point) { return point.second == 1.0f - 2 * point.first; }));

    std::cout << "All tests passed!" << std::endl;
}

void addNoiseToLine (std::vector<std::pair<float, float>>& line, float lowThreshold, float highThreshold)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution (lowThreshold, highThreshold);
    auto rand = std::bind (distribution, generator);
    std::for_each (line.begin(), line.end(), [&rand] (auto& point) { point.second += rand(); });
}

std::vector<std::pair<float, float>> getnoisyLine (float points)
{
    float bias = 1.0f;
    float weight = 2.0f;
    float startX = 0.0f;
    float endX = 10.0f;
    auto line = getLine (bias, weight, startX, endX, points);
    addNoiseToLine (line, -0.5f, 0.5f);

    return line;
}

torch::nn::Linear makeModel (int in, int out)
{
    auto net = torch::nn::Linear (in, out);
    net->to (torch::kCPU);

    for (const auto& param : net->parameters())
    {
        std::cout << "Parameter: " << param << std::endl;
    }
    return net;
}

torch::Tensor vec2tensor (const std::vector<float>& vec)
{
    torch::Tensor tensor = torch::empty ({ static_cast<int64_t> (vec.size()), 1 }, torch::kFloat);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        tensor[i][0] = vec[i];
    }
    return tensor;
}

int main()
{
    // int count = 1000;
    // auto line = getnoisyLine (count);
    //
    // std::cout << "Creating a simple model..." << std::endl;
    // int in { 1 }, out { 1 };
    // auto net = makeModel (in, out);
    //
    // auto optimizer = torch::optim::SGD (net->parameters(), /*lr=*/0.01);
    //
    // std::vector<float> xs = std::vector<float> (count);
    // std::transform (line.begin(), line.end(), xs.begin(), [] (const auto& x) { return x.first; });
    // auto input = vec2tensor (xs);
    //
    // std::vector<float> ys = std::vector<float> (count);
    // std::transform (line.begin(), line.end(), ys.begin(), [] (const auto& y) { return y.second; });
    // auto target = vec2tensor (ys);
    //
    // float lossValue = 1000.0f;
    // int cnt = 0;
    // while (lossValue > 0.5f)
    // {
    //     cnt++;
    //     auto output = net->forward (input);
    //
    //     torch::Tensor loss = torch::mse_loss (output, target);
    //     lossValue = loss.item<float>();
    //
    //     optimizer.zero_grad();
    //     loss.backward();
    //     optimizer.step();
    // }
    //
    // std::cout << "Training completed in " << cnt << " iterations with final loss: " << lossValue << std::endl;

    NeuralNetwork net (1, 2);
    for (int i = 0; i < 1000; ++i)
    {
        net.addTrainingData ({ i / 10.0f }, { i / 5.0f, i / 3.0f });
    }
    net.runTraining (1000000);
    return 0;
}
