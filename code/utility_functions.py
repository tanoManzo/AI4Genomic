## create a set of fuctions to be used in the pipeline
import numpy as np
import pandas as pd
import importlib

# Transformers library for pre-trained models and training utilities
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModel, 
    AutoConfig, 
    TrainingArguments, 
    Trainer, 
    BertForSequenceClassification
)

from transformers.models.bert.configuration_bert import BertConfig
import torch

# Function to get predictions in batches
def get_predictions_raw(models_tokenizers_dict, seq_ref, seq_alt, device="cuda", batch_size=32):
    models_predictions = {}

    def tokenize_in_batches(sequence, tokenizer, max_length=512, batch_size=32):
        tokens = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        for i in range(0, tokens['input_ids'].size(0), batch_size):
            yield {k: v[i:i+batch_size].to(device) for k, v in tokens.items()}

    for model_name, item in models_tokenizers_dict.items():
        model = item['model'].to(device).eval()
        tokenizer = item['tokenizer']

        print(f"Processing model: {model_name}")

        outputs_ref = []
        outputs_alt = []

        # Process in batches
        for inputs_ref in tokenize_in_batches(seq_ref, tokenizer, batch_size=batch_size):
            with torch.no_grad():
                batch_outputs_ref = model(**inputs_ref).logits.cpu()
                outputs_ref.append(batch_outputs_ref)
            torch.cuda.empty_cache()  # Clear memory after each batch

        for inputs_alt in tokenize_in_batches(seq_alt, tokenizer, batch_size=batch_size):
            with torch.no_grad():
                batch_outputs_alt = model(**inputs_alt).logits.cpu()
                outputs_alt.append(batch_outputs_alt)
            torch.cuda.empty_cache()  # Clear memory after each batch

        # Concatenate all batch results
        outputs_ref = torch.cat(outputs_ref, dim=0)
        outputs_alt = torch.cat(outputs_alt, dim=0)

        # Store results in CPU memory
        models_predictions[model_name] = {'ref': outputs_ref, 'alt': outputs_alt}

        # Free GPU memory by moving model to CPU and clearing cache
        model.to("cpu")
        torch.cuda.empty_cache()

    return models_predictions


def extract_SNPsequences_from_df(data_df, chrom2seq, length_bp=999):
    """
    Process sequences from a DataFrame and extract reference and alternative sequences.

    Parameters:
        mpra_df (pd.DataFrame): DataFrame containing chromosome, position, alt, and p-value columns.
        chrom2seq (dict): Dictionary mapping chromosomes to sequence data.
        length_bp (int): Length of the sequence to extract centered around each position.

    Returns:
        tuple: A tuple containing three lists:
            - seq_ref (list): List of reference sequences.
            - seq_alt (list): List of alternative sequences.
            - seq_val (list): List of values.
    """


    seq_ref = []
    seq_alt = []
    

    # Iterate over the DataFrame rows
    for idx, row in data_df.iterrows():
        chromosome = f"{row['Chromosome']}"
        abspos = row['Position']
        
        # Calculate the start and end positions for the sequence extraction
        start_pos = abspos - (length_bp // 2)-1
        end_pos = abspos + (length_bp // 2)  # Add 1 to ensure the length is exactly 1000 bp
        
        # Extract the sequence from the chromosome data
        seq = str(chrom2seq[chromosome][start_pos:end_pos])
        if len(seq) != length_bp:
            raise ValueError(f"Extracted sequence length {len(seq)} does not match the expected length {length_bp}.")
        
        half_len = len(seq) // 2

        #seq_ref.append(seq)
        seq_ref.append(f"{seq[:half_len]}{row['Reference']}{seq[half_len + 1:]}")
        

        # Create the alternative sequence by replacing the middle base with 'Alt'
        seq_alt.append(f"{seq[:half_len]}{row['Alternative']}{seq[half_len + 1:]}")

        if seq[half_len]!= row['Reference'] and seq[half_len]!= row['Alternative']:
            print(f"Warning Nucleaotide does NOT matched Ref or Alt ({seq[half_len]}) at index {idx}. Provided Ref and Alt: {row['Reference']},{row['Alternative']} ")

    data_df['Seq_Reference'] = seq_ref
    data_df['Seq_Alternative'] = seq_alt
    return data_df

def data_preprocessing_experimental_result(type_data, name_data, dataset_path):
    """
    Preprocess experimental data based on the specified type and dataset.
    
    Parameters:
        type_data (str): Type of the data ('raQTL' or 'mpra').
        name_data (str): Name of the dataset.
        dataset_path (str): Path to the dataset.
        
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    updated_data_df = pd.DataFrame()
    path_file = dataset_path
    old_data_df = pd.read_csv(path_file, sep='\t' if type_data == 'raQTL' else ',')

    if type_data == 'raQTL':
        updated_data_df = process_raQTL_data(old_data_df, name_data)
    elif type_data == 'mpra':
        updated_data_df = process_mpra_data(old_data_df, name_data)
    
    return updated_data_df


def process_raQTL_data(old_data_df, name_data):
    """
    Process raQTL data.
    """
    updated_data_df = pd.DataFrame()
    updated_data_df['Chromosome'] = old_data_df['chr']
    updated_data_df['Position'] = old_data_df['SNPabspos']
    updated_data_df['Reference'] = old_data_df['ref']
    updated_data_df['Alternative'] = old_data_df['alt']
    updated_data_df['SNP_id'] = old_data_df['SNP_ID']
    
    type_cell = 'hepg2' if 'hepg2' in name_data else 'k562'
    updated_data_df['Value_Ratio'] = old_data_df[f'{type_cell}.alt.mean'] / old_data_df[f'{type_cell}.ref.mean']
    updated_data_df['Value_Diff'] = old_data_df[f'{type_cell}.alt.mean'] - old_data_df[f'{type_cell}.ref.mean']
    updated_data_df['Value_Ratio_log2'] = np.log2(updated_data_df['Value_Ratio'])
        
    updated_data_df['Value_Pvalue_signed'] = -np.log10(old_data_df[f'{type_cell}.wilcox.p.value']) * \
                                             np.sign(updated_data_df['Value_Diff'])
    updated_data_df['P_value'] = old_data_df[f'{type_cell}.wilcox.p.value']
    
    return updated_data_df


def process_mpra_data(old_data_df, name_data):
    """
    Process MPRA data for different datasets.
    """
    updated_data_df = pd.DataFrame()

    if 'GSE87711' in name_data:
        updated_data_df['Chromosome'] = old_data_df['chr'].apply(lambda x: f'chr{x}')
        updated_data_df['Position'] = old_data_df['pos']
        updated_data_df['Reference'] = old_data_df['ref']
        updated_data_df['Alternative'] = old_data_df['alt']
        updated_data_df['Value_Ratio'] = old_data_df['CTRL.fc(log2)']
        updated_data_df['Value_Diff'] = old_data_df['CTRL.padj'] - old_data_df['CTRL.mut.padj']
        updated_data_df['Value_Ratio_log2'] = np.log2(old_data_df['CTRL.fc(log2)'])
        updated_data_df['Value_Pvalue_signed'] = -np.log10(old_data_df['CTRL.mut.p']) * \
                                                 np.sign(updated_data_df['Value_Ratio_log2'])
      
            
        updated_data_df['P_value'] = old_data_df['CTRL.mut.p']
        updated_data_df['SNP_id'] = old_data_df['dbSNP']

    elif 'SORT1' in name_data:
        updated_data_df['Chromosome'] = old_data_df['Chromosome'].apply(lambda x: f'chr{x}')
        updated_data_df['Position'] = old_data_df['Position']
        updated_data_df['Reference'] = old_data_df['Ref']
        updated_data_df['Alternative'] = old_data_df['Alt']
        updated_data_df['Value_Ratio_log2'] = old_data_df['VariantExpressionEffect (log2)']   
        updated_data_df['Value_Pvalue_signed'] = -np.log10(old_data_df['P-value']) * \
                                                 np.sign(updated_data_df['Value_Ratio_log2'])
        
        updated_data_df['P_value'] = old_data_df['P-value']
        updated_data_df['SNP_id'] = ''

    elif 'GSE68331' in name_data:
        updated_data_df['Chromosome'] = old_data_df['chr3']
        updated_data_df['Position'] = old_data_df['Pos']
        updated_data_df['Reference'] = old_data_df['Allele0']
        updated_data_df['Alternative'] = old_data_df['Allele1']
        #updated_data_df['Value_Ratio'] = old_data_df['effect']
        
        updated_data_df['Value_Ratio_log2'] = np.log2(old_data_df['effect'])
        updated_data_df['Value_Pvalue_signed'] = -np.log10(old_data_df['P']) * \
                                                 np.sign(np.log2(old_data_df['effect']))
        updated_data_df['P_value'] = old_data_df['P']
        updated_data_df['SNP_id'] = old_data_df['id']

    elif 'NPC_SNP' in name_data:
        updated_data_df['Chromosome'] = old_data_df['Chromosome']
        updated_data_df['Position'] = old_data_df['Central variant position (hg19)']
        updated_data_df['Reference'] = old_data_df['Archaic sequence sequence'].apply(lambda x: x[99])
        updated_data_df['Alternative'] = old_data_df['Modern sequence sequence'].apply(lambda x: x[99])
        #updated_data_df['Value_Ratio'] = old_data_df['Differential activity log2(fold-change) - modern vs archaic - NPC']
        updated_data_df['Value_Ratio_log2'] = old_data_df['Differential activity log2(fold-change) - modern vs archaic - NPC']
            
        updated_data_df['Value_Pvalue_signed'] = -np.log10(old_data_df['Differential activity P-value - NPC']) * \
                                                 np.sign(updated_data_df['Value_Ratio_log2'])
        updated_data_df['P_value'] = old_data_df['Differential activity P-value - NPC']
        updated_data_df['SNP_id'] = ''

    elif 'Hela' in name_data:
        updated_data_df['Chromosome'] = old_data_df['chromosome (hg19)']
        updated_data_df['Position'] = old_data_df['coordinate (hg19)']
        updated_data_df['Reference'] = old_data_df['Reference']
        updated_data_df['Alternative'] = old_data_df['Substitution']
        updated_data_df['Value_Ratio'] = old_data_df['HeLa effect size']
        updated_data_df['Value_Pvalue_signed'] = -np.log10(old_data_df['HeLa P-Value']) * \
                                                 np.sign(updated_data_df['Value_Ratio'])
        
        updated_data_df['Value_Ratio_log2'] = old_data_df['HeLa effect size']
        updated_data_df['P_value'] = old_data_df['HeLa P-Value']
        updated_data_df['SNP_id'] = old_data_df['Context']

    return updated_data_df


def load_models_and_tokenizers(models_names, bios_id, ft_model_type):
    """
    Load models and their associated tokenizers based on model names, bios_id, and fine-tuning type.

    Parameters:
    - models_names (list): List of model names to load.
    - bios_id (str): Identifier for the biological dataset.
    - ft_model_type (str): Type of fine-tuning applied.

    Returns:
    - dict: A dictionary containing loaded models and tokenizers.
    """
    models_tokenizers_dict = {}

    for model_name in models_names:
        model_ckpt = f"tanoManzo/{model_name}_ft_{bios_id}_{ft_model_type}"
        print(f"Loading model and tokenizer for: {model_ckpt}")

        try:
            # Check model type and load appropriate model and tokenizer
            if 'dnabert2' in model_ckpt:
                model = BertForSequenceClassification.from_pretrained(model_ckpt, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
            
            elif 'Geneformer' in model_ckpt:
                # Special case for Geneformer
                tokenizer = AutoTokenizer.from_pretrained(
                    'tanoManzo/Geneformer_ft_Hepg2_1kbpHG19_DHSs_H3K27AC'
                )
                model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

            elif 'gena-' in model_ckpt:
                # Special handling for Gena models
                model = AutoModel.from_pretrained(model_ckpt, trust_remote_code=True)
                module_name = model.__class__.__module__

                # Determine appropriate classification head
                if 'bigbird' in model_ckpt:
                    cls = getattr(importlib.import_module(module_name), 'BigBirdForSequenceClassification')
                else:
                    cls = getattr(importlib.import_module(module_name), 'BertForSequenceClassification')

                model = cls.from_pretrained(model_ckpt, num_labels=2)
                tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
            
            else:
                # Default case for other models
                tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, trust_remote_code=True)

            # Store the loaded model and tokenizer
            models_tokenizers_dict[f"{model_name}_ft_{bios_id}"] = {'model': model, 'tokenizer': tokenizer}

        except Exception as e:
            print(f"Error loading {model_ckpt}: {e}")

    return models_tokenizers_dict

