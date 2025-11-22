class EarlyStopping:
    def __init__(self, patience=15, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f" Count：{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
