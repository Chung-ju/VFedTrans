class FedRepresentationLearning():
    def __init__(self, frl_model, params) -> None:
        self.frl_model = frl_model
        self.model_params = params['model_params']
        self.exp_params = params['exp_params']

    def training(self, **kwargs):
        if self.model_params['name'] == 'FedSVD':
            self.frl_model.load_data(kwargs['X_shared'])
            self.frl_model.learning()
            Xs_fed = self.frl_model.get_fed_representation()
        elif self.model_params['name'] == 'VFedPCA':
            X_task, X_data = kwargs['X_task'], kwargs['X_data']
            Xs_fed = self.frl_model.fed_representation_learning(
                self.exp_params,
                kwargs['X_shared'],
                [X_task[X_task.shape[0]-kwargs['X_shared'].shape[0]:, :], X_data[X_data.shape[0]-kwargs['X_shared'].shape[0]:, :]])
        
        return Xs_fed