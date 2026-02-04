import argparse
import multiprocessing
from functools import partial
from typing import List

def generate_kmer_str_with_overlap(sequence:str, kmer:int=3) -> str:
    '''
        Construct k-mers with overlap from the original DNA sequence

        sequence: str 
            sequence of nucleotides to convert to k-mers
        kmer : int
            Number for k-mers (default=3)
    '''
    return ' '.join([sequence[i:i+kmer] for i in range(len(sequence) - kmer + 1)])

def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])

def process_chunk(chunk, k, seq_len, valid_chromosomes):
    """Processes a chunk of the reference file."""
    cur_line = ""
    collect_data = False
    results = []
    
    for line in chunk:
        line = line.strip().upper()
        
        if line.count("N") > 0:
            cur_line = ""
            continue
        elif line.count(">") > 0:
            chromosome = line.split(">")[1]
            collect_data = chromosome in valid_chromosomes
            cur_line = ""
            continue
        
        if collect_data:
            cur_line += line
            while len(cur_line) >= seq_len:
                new_line = cur_line[:seq_len]
                cur_line = cur_line[seq_len:]
                sentence = generate_kmer_str_with_overlap(new_line, kmer=k)
                results.append(sentence)
    
    return results

def collapse_methylation(seq:List) -> int:
    '''
    Computes aggregated methylation tag based on the tags applied to individual nucleotides in a given sequence seq
    '''
    if not seq:
        return 2 # Other for empty sequences
    if(any(x==1 for x in seq)):
        return 1 #  Methylated cytosine at CpG context
    elif(any(x==0 for x in seq)):
        return 0 #  Unmethylated cytosine at CpG context
    else:
        return 2 #  Other

def pretrain_data_preprocess(f_ref: str, k: int = 3, seq_len: int = 510, f_output: str = None, num_cores:int = None) -> None:
    '''
        Parallel version of generating N bp length k-mers sequence from reference genome as pretrain data.

        f_ref : str
            Path to the reference fasta file
        k : int
            Number for k-mers (default=3)
        seq_len: int
            Base-pair length of generated sequences (default=510)
        f_output : str
            Path to the output file, an appropriate name 
            will be automatically assigned if not given
    '''
    # TODO adapt to all models by allowing argument k = -1  
    # Get number of CPU cores and use 90% of them, but not less than one
    avaliable_cpus = multiprocessing.cpu_count()
    if num_cores is None:
        num_cores = max(1, int(avaliable_cpus * 0.9))
    else:
        assert num_cores <= avaliable_cpus, f"Number of cores should not be bigger than total number of cores avaliable: {avaliable_cpus}" 
    
    # Assign default output file name if not provided
    if f_output is None:
        f_output = f_ref.replace("Raw", "Refined") + f"_{k}mers_seqlen{seq_len}.txt"
    
    valid_chromosomes = ["CHR" + str(i) for i in range(1, 23)] + ["CHRX", "CHRY"]
    
    # Read the file in chunks and process in parallel
    with open(f_ref, "r") as fp_ref:
        lines = fp_ref.readlines()
    
    # Split the file into chunks for parallel processing
    chunk_size = len(lines) // num_cores
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    # Create a partial function with fixed arguments for parallel processing
    process_partial = partial(process_chunk, k=k, seq_len=seq_len, valid_chromosomes=valid_chromosomes)
    
    # Set up multiprocessing pool and process chunks in parallel
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_partial, chunks)
    
    # Flatten the results and write to the output file
    with open(f_output, "w") as fp_out:
        for result in results:
            for sentence in result:
                fp_out.write(sentence + "\n")

    print(f"Processing completed. Output saved to {f_output}")

# TODO: Cover the actual standalone application in scripts instead (not part of the package)
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--kmer",
#         default=1,
#         type=int,
#         help="K-mer",
#     )
#     parser.add_argument(
#         "--length",
#         default=10000,
#         type=int,
#         help="Length of the sampled sequence",
#     )
#     parser.add_argument(
#         "--file_path",
#         default=None,
#         type=str,
#         help="The path of the file to be processed",
#     )
#     parser.add_argument(
#         "--output_path",
#         default=None,
#         type=str,
#         help="The path of the processed data",
#     )
#     parser.add_argument(
#         "--num_cores",
#         default=None,
#         type=str,
#         help="The number of cores used for processing",
#     )
#     args = parser.parse_args()
#     return args
    
# if __name__ == "__main__":
#     args = parse_args()
#     pretrain_data_preprocess(args.file_path, k=args.kmer, 
#                              seq_len=args.length, f_output=args.output_path, num_cores = args.num_cores)

