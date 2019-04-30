import Base: length, iterate
using Random
using CSV
using PyCall

VOCABFILE = "bert-base-uncased-vocab.txt"
NUM_CLASSES = 2

token2int = Dict()
f = open(VOCABFILE) do file
    lines = readlines(file)
    for (i,line) in enumerate(lines)
        token2int[line] = i
    end
end
int2token = Dict(value => key for (key, value) in token2int)
VOCABSIZE = length(token2int)

include("preprocess.jl")

mutable struct ClassificationData
    input_ids
    input_mask
    segment_ids
    labels
    batchsize
    ninstances
    shuffled
end

function ClassificationData(input_file, token2int; batchsize=8, shuffled=true, seq_len=64)
    input_ids = []
    input_mask = []
    segment_ids = []
    labels = []
    (x, labels) = read_and_process(input_file, token2int)
    for i in 1:length(x)
        if length(x[i]) >= seq_len
            x[i] = x[i][1:seq_len]
            mask = Array{Int64}(ones(seq_len))
        else
            mask = Array{Int64}(ones(length(x[i])))
            append!(x[i], fill(1, seq_len - length(x[i]))) # 1 is for "[PAD]"
            append!(mask, fill(0, seq_len - length(mask))) # 0's vanish with masking operation
        end
        push!(input_ids, x[i])
        push!(input_mask, mask)
        push!(segment_ids, Array{Int64}(ones(seq_len)))
    end
    ninstances = length(input_ids)
    return ClassificationData(input_ids, input_mask, segment_ids, labels, batchsize, ninstances, shuffled)
end


function length(d::ClassificationData)
    d, r = divrem(d.ninstances, d.batchsize)
    return r == 0 ? d : d+1
end

function iterate(d::ClassificationData, state=ifelse(d.shuffled, randperm(d.ninstances), 1:d.ninstances))

    state === nothing && return nothing

    if length(state) > d.batchsize
        new_state = state[d.batchsize+1:end]
        input_ids = hcat(d.input_ids[state[1:d.batchsize]]...)
        input_mask = hcat(d.input_mask[state[1:d.batchsize]]...)
        segment_ids = hcat(d.segment_ids[state[1:d.batchsize]]...)
        labels = hcat(d.labels[state[1:d.batchsize]]...)
    else
        new_state = nothing
        input_ids = hcat(d.input_ids[state]...)
        input_mask = hcat(d.input_mask[state]...)
        segment_ids = hcat(d.segment_ids[state]...)
        labels = hcat(d.labels[state]...)
    end
    
    return ((input_ids, input_mask, segment_ids, labels), new_state)
end

mutable struct ClassificationData2
    input_ids
    input_mask
    segment_ids
    labels
    batchsize
    ninstances
    shuffled
end


function ClassificationData2(input_file; batchsize=8, shuffled=true, seq_len=64)
    input_ids = []
    input_mask = []
    segment_ids = []
    labels = []
    f = open(input_file)
    tmp = split.(readlines(f), "\t")
    for i in 1:length(tmp)
        instance = eval.(Meta.parse.(tmp[i]))
        push!(input_ids, (instance[1] .+ 1)[1:seq_len])
        push!(input_mask, instance[2][1:seq_len])
        push!(segment_ids, (instance[3] .+ 1)[1:seq_len])
        push!(labels, (instance[4] + 1))
    end
    ninstances = length(input_ids)
    return ClassificationData2(input_ids, input_mask, segment_ids, labels, batchsize, ninstances, shuffled)
end


function length(d::ClassificationData2)
    d, r = divrem(d.ninstances, d.batchsize)
    return r == 0 ? d : d+1
end

function iterate(d::ClassificationData2, state=ifelse(d.shuffled, randperm(d.ninstances), 1:d.ninstances))

    state === nothing && return nothing

    if length(state) > d.batchsize
        new_state = state[d.batchsize+1:end]
        input_ids = hcat(d.input_ids[state[1:d.batchsize]]...)
        input_mask = hcat(d.input_mask[state[1:d.batchsize]]...)
        segment_ids = hcat(d.segment_ids[state[1:d.batchsize]]...)
        labels = hcat(d.labels[state[1:d.batchsize]]...)
    else
        new_state = nothing
        input_ids = hcat(d.input_ids[state]...)
        input_mask = hcat(d.input_mask[state]...)
        segment_ids = hcat(d.segment_ids[state]...)
        labels = hcat(d.labels[state]...)
    end
    
    return ((input_ids, input_mask, segment_ids, labels), new_state)
end

include("model.jl")

# Embedding Size, Vocab Size, Intermediate Hidden Size, Max Sequence Length, Sequence Length, Num of Segments, Num of Heads in Attention, Num of Encoders in Stack, Batch Size, Matrix Type, General Dropout Rate, Attention Dropout Rate, Activation Function 
config = Config(768, 30522, 3072, 512, 64, 2, 12, 12, 10, KnetArray{Float32}, 0.1, 0.1, gelu)

dtrn = ClassificationData2("../project/sst-train.tsv", batchsize=config.batchsize, seq_len=config.seq_len)
ddev = ClassificationData2("../project/sst-dev.tsv", batchsize=config.batchsize, seq_len=config.seq_len)
dtst = ClassificationData2("../project/sst-test.tsv", batchsize=config.batchsize, seq_len=config.seq_len)

model = BertClassification(config, NUM_CLASSES)

@pyimport torch
torch_model = torch.load("/scratch/users/omutlu/dl_course/project/pytorch_model.bin")

model = load_from_torch_base(model, config.num_encoder, config.atype, torch_model)

function accuracy2(model, dtst)
    true_count = 0
    all_count = 0
    for (x, attention_mask, segment_ids, y) in dtst
        probs = model(x, segment_ids, attention_mask=attention_mask)
        preds = map(x -> x[1], argmax(Array{Float32}(probs),dims=1))
        true_count += sum(y .== preds)
        all_count += length(y)
    end
    return true_count/all_count
end

function initopt!(model, optimizer="Adam()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end

function mytrain!(model, dtrn, ddev, best_acc)
    losses = []
    for (k, (x, attention_mask, segment_ids, labels)) in enumerate(dtrn)
        J = @diff model(x, segment_ids, labels, attention_mask=attention_mask)
        for par in params(model)
            g = grad(J, par)
            update!(value(par), g, par.opt)
        end
        push!(losses, value(J))
        if k % 5000 == 0
            acc = accuracy2(model, ddev)
            println("Accuracy at $k iteration : ", acc)
            if acc > best_acc
                best_acc = acc
                println("Saving...")
                Knet.save("model_bert.jld2", "model", model)
            end
        end
    end
    return Knet.mean(losses)
end

initopt!(model)

#trnloss = [model(dtrn)]
#devloss = [model(ddev)]
devloss = []
best_acc = 0.0
for epoch in 1:30
    global best_acc
    println("Epoch : ", epoch)
    lss = mytrain!(model, dtrn, ddev, best_acc)
    #push!(trnloss, model(dtrn))
    println("Training loss at $epoch epoch is : $lss")
    #=
    push!(devloss, model(ddev))
    println("Dev loss is : $devloss[epoch]")
    acc = accuracy2(model, ddev)
    println("Accuracy : ", acc)
    if acc > best_acc
        best_acc = acc
        println("Saving...")
        Knet.save("model_bert.jld2", "model", model)
    end
    =#
end

model = Knet.load("model_bert.jld2", "model")
result = accuracy2(model, dtst)
println("Test accuracy is : $result")