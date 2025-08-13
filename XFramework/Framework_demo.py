import pandas as pd
import numpy as np
import time
import os
from XFramework.Framework import EMMM, ModelWrapper
from XFramework.DataManager import DataManager


class DetectionExplanationHistory: 
    def __init__(self):
        self.dia = []
        self.dia_DA = []
        self.utt_detection = []
        self.da_detection = []
        self.dia_detection = []
        self.utt_expl = None
        self.da_expl = None
        self.utt_aggregation_plots = []
        self.n_turn = 0
        
    def update(self, utt, utt_DA, utt_detection, da_detection, turn_lv_utt_expl, turn_lv_da_expl):
        self.dia.append(utt)
        self.dia_DA.append(utt_DA)
        self.utt_detection.append(utt_detection)
        self.da_detection.append(da_detection)
        
        if self.utt_expl is None: 
            self.utt_expl = turn_lv_utt_expl
        else: 
            self.utt_expl = pd.concat([self.utt_expl, turn_lv_utt_expl], ignore_index=True)
        
        if self.da_expl is None: 
            self.da_expl = turn_lv_da_expl
        else: 
            self.da_expl = pd.concat([self.da_expl, turn_lv_da_expl], ignore_index=True)
        
        self.n_turn += 1


class EMMM_Demo:
    DEFAULT_DIA_NO = 0
    DEFAULT_LABEL = 0

    def __init__(self, framework:EMMM, sensitive_info_masker, da_extractor, report_generator, demo_id, log=True):
        self.framework=framework # trained framework instance (keep track of the framework settings)
        self.utt_detection_model = None
        self.da_detection_model = None
        self.dia_detection_model = None
        self.prepare_detection_explanation_models() 

        self.sensitive_info_masker = sensitive_info_masker
        self.da_extractor = da_extractor
        self.report_generator = report_generator

        self.report_generator.demo_id = demo_id
        self.historical_result = DetectionExplanationHistory()

        self.artefact_folder = os.path.join(self.framework.save_dir, 'demo')
        os.makedirs(self.artefact_folder, exist_ok=True)
        self._clear_artefacts()

        self.log=log
        
    def _clear_artefacts(self): 
        for filename in os.listdir(self.artefact_folder):
            file_path = os.path.join(self.artefact_folder, filename)
            if os.path.isfile(file_path): 
                os.remove(file_path)

    def reset_demo(self, new_demo_id): 
        # set new demo_id and clear history
        self.report_generator.demo_id = new_demo_id
        self.historical_result = DetectionExplanationHistory()
        self._clear_artefacts()
    
    def prepare_detection_explanation_models(self): 
        tokenizer, model, model_wrapper = self.framework.load_detection_model(dimension='da')
        self.da_detection_model = {'tokenizer': tokenizer, 'model': model, 'model_wrapper': model_wrapper}

        tokenizer, model, model_wrapper = self.framework.load_detection_model(dimension='utt')
        self.utt_detection_model = {'tokenizer': tokenizer, 'model': model, 'model_wrapper': model_wrapper}

        self.dia_detection_model = ModelWrapper(self.framework.load_dia_lv_model()).predict
        
    def utt_explainer(self, dataset):
        return self.framework.extract_explanation_turn_lv_utt_dataset(dataset, 
                    self.utt_detection_model['model'], self.utt_detection_model['tokenizer'], self.utt_detection_model['model_wrapper'], 
                    generate_data_path=os.path.join(self.framework.save_dir, 'demo', 'demo_utt_expl'), 
                    predictions = None, explanation_df = None, log_save_results=False)

    def da_explainer(self, dataset):
        return self.framework.extract_explanation_turn_lv_da_dataset(dataset, 
                    self.da_detection_model['model'], self.da_detection_model['tokenizer'], self.da_detection_model['model_wrapper'], 
                    generate_data_path=os.path.join(self.framework.save_dir, 'demo', 'demo_da_expl'), 
                    predictions = None, explanation_df = None, log_save_results=False)

    def mask_sensitive_info(self, utt): 
        return self.sensitive_info_masker(utt)

    def preprocess(self, utt): 
        # mask sensitive information
        return self.mask_sensitive_info(utt)

    def extract_DA(self, utt): 
        return self.da_extractor(utt, self.historical_result.dia)
        
    def detect_utt(self, dataset): 
        return self.utt_detection_model['model_wrapper'].predict(dataset)

    def detect_da(self, dataset): 
        return self.da_detection_model['model_wrapper'].predict(dataset)

    def detect_dia(self): 
        dia_lv_df = pd.DataFrame({'dia_no': [EMMM_Demo.DEFAULT_DIA_NO], 
                                'label':[EMMM_Demo.DEFAULT_LABEL], 
                                'masked_da':[list(self.historical_result.da_expl['masked_sample'].values)], 
                                'masked_utt':['\n'.join(list(self.historical_result.utt_expl['masked_sample']))]})
        dia_lv_dataset = DataManager(dia_lv_df, 
                                None, 
                                None, 
                                level='dialogue',
                                model_name=self.framework.turn_lv_da_model['model_name'],
                                dataset_preprocess=DataManager.multimodal_da_utt, 
                                dataset_initialize = 'separate', 
                                batch_size=8, max_len=DataManager.DIA_MAX_LEN).train_dataloader.dataset
        if self.log: 
            print(self.utt_detection_model['tokenizer'].decode(dia_lv_dataset['da_input_ids'][0]), flush=True)
            print(self.utt_detection_model['tokenizer'].decode(dia_lv_dataset['utt_input_ids'][0]), flush=True)
        return self.dia_detection_model(dia_lv_dataset)

    def detect_explain_utt(self, sys_utt, utt): 
        if self.log: 
            print("\n\n"+"="*200+"\n", flush=True)
            print('original utterance:', utt, flush=True)

        # assumed preprocessing of the utterance - e.g. masking sensitive information
        utt = self.preprocess(utt)
        if self.log: 
            print('preprocessed utterance:', utt, flush=True)

        # DETECTION + EXPLANATION PIPELINE
        start_time = time.time() 
        # extract DA
        self.historical_result.dia.append(sys_utt)
        utt_DA = self.extract_DA(utt)

        da_extract_end = time.time()

        # package into a dataset
        dataset = pd.DataFrame({'dia_no': [EMMM_Demo.DEFAULT_DIA_NO], 'dia': [utt], 'dialogue_act_info_removed': [[utt_DA]], 'label': [EMMM_Demo.DEFAULT_LABEL], 'turn_no':[[self.historical_result.n_turn]]})
        utt_dataset = DataManager(dataset, None, None, 
                                   tokenizer = self.utt_detection_model['tokenizer'], 
                                   dataset_preprocess=DataManager.utt_only, batch_size=16, max_len=DataManager.UTT_TURN_MAX_LEN).train_dataloader.dataset
        if self.log: 
            print(self.utt_detection_model['tokenizer'].decode(utt_dataset['input_ids'][0]), flush=True)
        da_dataset = DataManager(dataset, None, None, 
                                   tokenizer = self.da_detection_model['tokenizer'], 
                                   dataset_preprocess=DataManager.da_turn, batch_size=16, max_len=DataManager.DA_TURN_MAX_LEN).train_dataloader.dataset
        if self.log: 
            print(self.da_detection_model['tokenizer'].decode(da_dataset['input_ids'][0]), flush=True)

        # turn-lv detection
        turn_lv_utt_detection = self.detect_utt(utt_dataset)[0].item()
        turn_lv_da_detection = self.detect_da(da_dataset)[0].item()

        local_expl_start_time = time.time()
        
        # local explanation extraction
        turn_lv_utt_expl = self.utt_explainer(utt_dataset)
        turn_lv_da_expl = self.da_explainer(da_dataset)

        local_expl_end_time = time.time()

        # update historical result
        self.historical_result.update(utt, utt_DA, turn_lv_utt_detection, turn_lv_da_detection, turn_lv_utt_expl, turn_lv_da_expl)

        # dia-lv detection
        self.historical_result.dia_detection.append(self.detect_dia()[0].item())

        # report generation
        report_start_time = time.time()
        report = self.report_generator(self.historical_result)
        
        end_time = time.time()

        time_metric = {'framework_start': start_time,
                    'da_extract_end': da_extract_end,
                    'local_expl_start': local_expl_start_time,
                    'local_expl_end': local_expl_end_time,
                    'report_start': report_start_time, 
                    'framework_end': end_time, 
                    'da_extract time': da_extract_end - start_time, 
                    'turn-lv detection time': local_expl_start_time - da_extract_end,
                    'local expl time': local_expl_end_time - local_expl_start_time, 
                    'dia-lv detection time': report_start_time - local_expl_end_time,
                    'report_generation time': end_time - report_start_time,
                    'framework time': end_time - start_time}
        if self.log: 
            print('DA:', utt_DA, flush=True)
            print('turn_lv_utt_detection:', turn_lv_utt_detection, flush=True)
            print('turn_lv_da_detection:', turn_lv_da_detection, flush=True)
            print('turn_lv_utt_expl:', turn_lv_utt_expl['masked_sample'][0], flush=True)
            print('turn_lv_da_expl:', turn_lv_da_expl['masked_sample'][0], flush=True)
            print('dia_detection:', self.historical_result.dia_detection, flush=True)
            print(time_metric, flush=True)
            print("\n\n", flush=True)

        # remove artefacts (explanation files which will be reused if exist)
        self._clear_artefacts()

        return self.historical_result, time_metric
