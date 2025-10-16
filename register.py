import json
import re
import time
import pandas as pd
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from pyspark.sql import DataFrame as SparkDataFrame
from vllm import EngineArgs
from threading import Lock
from util.prompt import DEFAULT_SYSTEM_PROMPT
from util.vllm import vLLM
from util.utils import LLM, get_ordered_columns
from pandas import DataFrame
from util.utils import *
from string import Formatter
import os
from pyspark.sql.functions import col, window, expr, sum, avg, count, when, lit, unix_timestamp, date_format, pandas_udf

# from pyflink.datastream import StreamExecutionEnvironment
# from pyflink.table import StreamTableEnvironment, DataTypes, EnvironmentSettings
# from pyflink.table.udf import udf
# from pyflink.common.typeinfo import Types
# from pyflink.common import Row
# from pyflink.datastream.functions import MapFunction, ProcessFunction
# from pyflink.datastream.state import ValueStateDescriptor
import time
import random
import csv
import os
from threading import Thread

class ModelRegistry:
    _instance = None
    _lock = Lock()
    _model = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def initialize_model(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    print("Initializing global model...")
                    engine_args = EngineArgs(
                        model="/data/models/Qwen2.5-Coder-7B-Instruct",
                        max_num_seqs=1024
                    )
                    self._model = vLLM(
                        engine_args=engine_args,
                        base_url="http://localhost:8002/v1"
                    )
                    self._tokenizer = get_tokenizer()
                    self._initialized = True
                    print("Global model initialized successfully.")

    @property
    def tokenizer(self):
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = get_tokenizer()
        return self._tokenizer

    @property
    def model(self) -> LLM:
        if not self._initialized:
            self.initialize_model()
        return self._model

# Global variables
global_spark = None
global_table_name = ""
processed_row = 0


def get_fields(user_prompt: str) -> List[str]:
    """Get the names of all the fields specified in the user prompt."""
    if not isinstance(user_prompt, str):
        raise ValueError("Expected a string for user_prompt, got: {}".format(type(user_prompt)))
    pattern = r"{(.*?)}"
    return re.findall(pattern, user_prompt)

def batchQuery(model: LLM, 
          prompt: str, 
          df: DataFrame, 
          system_prompt: str = DEFAULT_SYSTEM_PROMPT,
          guided_choice: List[str] = None
          ):


    df.drop_duplicates()
    # Returns a list of dicts, maintaining column order.
    records = df.to_dict(orient="records")
    outputs = model.execute_batch(
        fields=records,
        query=prompt,
        system_prompt=system_prompt,
        guided_choice=guided_choice
    )
    return outputs

# Pyspark UDF
# @pandas_udf(StringType())
# def llm_udf_v2(prompts: pd.Series, *dataFrames: pd.Series) -> pd.Series:
#     print("len of prompts", len(prompts))
#     outputs = []    
#     prompt = prompts.iloc[0]
#     # Process each Series in dataFrames
#     fields = get_fields(prompts.iloc[0])
#     if len(dataFrames) != len(fields):
#         raise ValueError(
#             f"Expected {len(fields)} context column(s) (for placeholders {fields}), "
#             f"but got {len(dataFrames)}."
#         )

#     # Extract fields dynamically from the user prompt
#     merged_df = pd.DataFrame({field: col.apply(lambda x: x.split(':', 1)[1].strip() if ':' in x else x) 
#                              for field, col in zip(fields, dataFrames)})
#     start_time = time.time()
#     # print("\n=== Before Reordering ===")
#     # print("Columns:", merged_df.columns.tolist())
#     # print("\nFirst 5 rows:")
#     # print(merged_df.head())

#     # merged_df, _ = QuickGreedy().reorder(
#     #         merged_df,
#     #         early_stop=solver_config["early_stop"],
#     #         row_stop=solver_config.get("row_stop", None),
#     #         col_stop=solver_config.get("col_stop", None),
#     #         col_merge=merged_cols,
#     #         one_way_dep=one_deps,
#     #         distinct_value_threshold=solver_config.get("distinct_value_threshold", default_distinct_value_threshold),
#     #     )
#     after_reorder_time = time.time()
#     #print(f"\nReordering time: {after_reorder_time - start_time:.4f} seconds")

#     # print("Columns:", merged_df.columns.tolist())
#     # print("\nFirst 5 rows:")
#     # print(merged_df.head())

#     before_query_time = time.time()
#     # Call the query function for batch execution using the global model
#     outputs = batchQuery(
#         model=model,
#         prompt=prompt,
#         df=merged_df,
#         system_prompt=DEFAULT_SYSTEM_PROMPT
#     )

#     end_time = time.time()
#     print(f"\nBatch query time: {end_time - before_query_time:.4f} seconds")
#     print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")
    
#     return pd.Series(outputs)

# Register the LLM UDF with Spark
def register_llm_udf():
    nondet_llm_udf = llm_udf_v2.asNondeterministic()
    global_spark.udf.register("LLM", nondet_llm_udf)


# class LLM(ProcessFunction):
#     def __init__(self, query):
#         # Get the global model instance
#         self.model = model_registry.model
#         self.query = query
#         self.base_url = "http://localhost:8002/v1"
#         self.model_name = is_server_running()
        

#     def generate_prompt(self, user_prompt: str, system_prompt: str) -> str:
#         messages = [
#             {"role": "user", "content": user_prompt},
#             {"role": "system", "content": system_prompt}
#         ]

#         successful_prompt_generation = False
#         while not successful_prompt_generation:
#             try:
#                 # Construct a prompt for the chosen model given OpenAI style messages.
#                 prompt = self.model.tokenizer.apply_chat_template(
#                     conversation=messages,
#                     tokenize=False,
#                     add_generation_prompt=True
#                 )
#             except Exception as e:
#                 if messages[0]["role"] == "system":
#                     # Try again without system prompt
#                     messages = messages[1:]
#                 else:
#                     raise e
#             else:
#                 successful_prompt_generation = True
        
#         return prompt

#     def process_element(self, row: Row, ctx: ProcessFunction.Context):
#         row_dict = row.as_dict()
#         try:
            
#             field_names = [fname for _, fname, _, _ in Formatter().parse(self.query) if fname]
#             prompt_data = {field: row_dict[field] for field in field_names}
#             user_prompt_template = f"{self.query} Given the following data:\n {prompt_data} \n answer the above query:"
#             prompts = [self.generate_prompt(user_prompt_template, DEFAULT_SYSTEM_PROMPT)]
#             #print(prompts[0])
#             #prompts = [user_prompt_template]
#             request_outputs = json.loads(post_http_request(self.model_name, prompts, temperature=0, api_url=(self.base_url + "/completions")).content)
#             #print("result: ", [choice['text'] for choice in request_outputs['choices']])

#         except KeyError as e:
#             print(f"Missing column in row data: {e}")


from typing import List, Optional
class SemanticExtension:
    def __init__(self, model: Optional[LLM] = None, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.model = model
        self._register_extension()
        
        
    
    def _extract_placeholders(self, template: str) -> List[str]:
        return re.findall(r'\{([^}]+)\}', template)
    

    def _register_extension(self):

        @pandas_udf(BooleanType())
        def _sem_filter_udf(prompts: pd.Series, *dataFrames: pd.Series) -> pd.Series:
            print("len of prompts", len(prompts))
            outputs = []    
            prompt = prompts.iloc[0]
            
            
            # Process each Series in dataFrames
            fields = get_fields(prompt)
            if len(dataFrames) != len(fields):
                raise ValueError(
                    f"Expected {len(fields)} context column(s) (for placeholders {fields}), "
                    f"but got {len(dataFrames)}."
                )
        

            # Extract fields dynamically from the user prompt
            merged_df = pd.DataFrame({field: col 
                                    for field, col in zip(fields, dataFrames)})

            for col_name in merged_df.columns:
                prefix = f"{col_name}: "
                if pd.api.types.is_string_dtype(merged_df[col_name]):
                    merged_df[col_name] = merged_df[col_name].str.replace(prefix, '', n=1)

            text_outputs = batchQuery(
                model=self.model,
                prompt=prompt,
                df=merged_df,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                guided_choice=["true", "false"]
            )
            
            # Convert string outputs to boolean directly in the UDF
            boolean_outputs = pd.Series([output.lower() == "true" for output in text_outputs])
            return boolean_outputs
            
        self.sem_filter_udf = _sem_filter_udf.asNondeterministic()

        def sem_filter_v2(dataframe, filter_template: str, output_col: str = "_keep_record") -> SparkDataFrame:
            system_prompt = self.system_prompt
            
            udf_args = [lit(filter_template)]

            fields = get_fields(filter_template)

            for col_name in fields:
                udf_args.append(col(col_name))

            temp_col = output_col if output_col not in dataframe.columns else f"__temp_{output_col}"
            schema = dataframe.schema
            filtered_df = dataframe.filter(self.sem_filter_udf(*udf_args))
            return filtered_df

        SparkDataFrame.sem_filter_v2 = sem_filter_v2
