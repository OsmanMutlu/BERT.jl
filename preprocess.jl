using CSV

function wordpiece_tokenize(token, dict)
    # This is a longest-match-first algorithm.
    out_tokens = []
    start = 1
    while start <= length(token)
        finish = length(token)
        final_token = ""
        for i in finish:-1:start
            # String Indexing Error for an unknown reason. Might be because of unicode chars.
            tkn = try
                start == 1 ? token[start:i] : string("##", token[start:i])
            catch
                ""
            end
            if tkn in keys(dict)
                final_token = tkn
                finish = i
                break
            end
        end
        if final_token == "" # if there is no match at all, assign unk token
            return ["[UNK]"]
        end
        push!(out_tokens, final_token)
        start = finish + 1
    end
    return out_tokens
end

# Not implemented yet
function process_punc(tokens)
    return tokens
end

function tokenize(text, dict)
    text = strip(text)
    if text == ""
        return []
    end
    tokens = split(text)
    tokens = process_punc(tokens)
    out_tokens = []
    for token in tokens
        append!(out_tokens, wordpiece_tokenize(token, dict))
    end
    return out_tokens
end

function convert_to_int_array(text, dict)
    tokens = tokenize(text, dict)
    out = Int[]
    for token in tokens
        if token in keys(dict)
            push!(out, dict[token])
        else
            push!(out, dict["[UNK]"])
        end
    end
    return out
end

function read_and_process(filename, dict)
    data = CSV.File(filename, delim="\t")
    x = Array{Int,1}[]
    y = Int8[]
    for i in data
        push!(x, convert_to_int_array(i.sentence, dict))
        push!(y, Int8(i.label + 1)) # negative 1, positive 2
    end
    
    # Padding to maximum
#     max_seq = findmax(length.(x))[1]
#     for i in 1:length(x)
#         append!(x[i], fill(1, max_seq - length(x[i]))) # 1 is for "[PAD]"
#     end
    
    return (x, y)
end