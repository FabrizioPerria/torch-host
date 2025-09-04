#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork (int inputSize, int outputSize) : numInputs (inputSize), numOutputs (outputSize)
{
    linearLayer1 = register_module ("linear1", torch::nn::Linear (numInputs, numOutputs));

    // auto sigmoid = torch::nn::Sigmoid();
    // sigmoid->to (torch::kCPU);
    // sigmoidLayer = register_module ("sigmoid", sigmoid);

    linearLayer2 = register_module ("linear2", torch::nn::Linear (numOutputs, numOutputs));

    // softmaxLayer = register_module ("softmax", torch::nn::Softmax (1));

    optimizer = std::make_unique<torch::optim::SGD> (parameters(), 0.01);
}

std::vector<float> NeuralNetwork::forward (const std::vector<float>& input)
{
    static thread_local std::vector<float> output (numOutputs);
    static thread_local torch::Tensor inputTensor = torch::empty ({ 1, numInputs }, torch::kFloat);

    std::memcpy (inputTensor.data_ptr<float>(), input.data(), numInputs * sizeof (float));

    torch::NoGradGuard no_grad;
    torch::Tensor outputTensor = forward (inputTensor);

    std::memcpy (output.data(), outputTensor.data_ptr<float>(), numOutputs * sizeof (float));

    return output;
}

void NeuralNetwork::addTrainingData (const std::vector<float>& input, const std::vector<float>& target)
{
    assert (input.size() == numInputs);
    assert (target.size() == numOutputs);
    torch::Tensor inT = torch::from_blob (const_cast<float*> (input.data()), numInputs, torch::kFloat).clone();
    torch::Tensor targetT = torch::from_blob (const_cast<float*> (target.data()), numOutputs, torch::kFloat).clone();
    trainingInputs.push_back (inT);
    trainingTargets.push_back (targetT);
}

void NeuralNetwork::runTraining (int epochs)
{
    if (trainingInputs.empty() || trainingTargets.empty())
    {
        std::cerr << "No training data available!" << std::endl;
        return;
    }
    torch::Tensor input = torch::cat (trainingInputs)
                              .reshape ({ static_cast<int64_t> (trainingInputs.size()), trainingInputs[0].sizes()[0] });
    torch::Tensor target = torch::cat (trainingTargets)
                               .reshape ({ static_cast<int64_t> (trainingTargets.size()), trainingTargets[0].sizes()[0] });

    double loss = 0.0;
    double pLoss = 1000000.0;
    double dLoss = 1000000.0;
    for (int i = 0; i < epochs; ++i)
    {
        optimizer->zero_grad();
        auto lossResult = torch::mse_loss (forward (input), target);
        loss = lossResult.item<float>();
        lossResult.backward();
        optimizer->step();
        dLoss = pLoss - loss;
        pLoss = loss;
        if (i > 0)
        {
            if (dLoss < 0.00001 && loss < 0.00001)
            {
                std::cout << "Training converged at epoch " << i << " with loss: " << loss << std::endl;
                break;
            }
        }
    }
    std::cout << "Training completed after " << epochs << " epochs with final loss: " << loss << std::endl;
}

torch::Tensor NeuralNetwork::forward (const torch::Tensor input)
{
    torch::Tensor out = linearLayer1->forward (input);
    out = torch::sigmoid (out);
    out = linearLayer2->forward (out);

    // auto out = linearLayer (input);
    // out = softmaxLayer (out);
    return out;
}
