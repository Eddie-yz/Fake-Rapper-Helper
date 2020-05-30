from RecognitionEncoder import deepSpeech
def contextTraining(train_loader, rEncoder, epochs, criterion):
    for epoch in range(epochs):
        rEncoder.train()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader):
            state.set_training_step(training_step=i)
            inputs, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)

            out, output_sizes = rEncoder(inputs, input_sizes)

            float_out = out.float()  # ensure float32 for loss
            loss = criterion(float_out, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_value = loss.item()
            
if __name__ == "__main__":
    rEncoder = load_model('librispeech_pretrained_v2.pth')
    device = torch.device("cuda")
    rEncoder = rEncoder.to(device)
    print(model.audio_conf)

    minibatch_size = 8
    freq_size = 16000
    max_seqlength = 100
    cur_seqlength = torch.LongTensor([90]*minibatch_size)
    inputs = torch.rand(minibatch_size, 1, freq_size, max_seqlength).to(device)
    