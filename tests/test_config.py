from cassava import config


def test_label_map_has_five_classes():
    assert len(config.LABEL_MAP) == 5
    assert set(config.LABEL_MAP.keys()) == {0, 1, 2, 3, 4}


def test_target_size_matches_label_map():
    assert config.TARGET_SIZE == len(config.LABEL_MAP)


def test_healthy_class_exists():
    assert "Healthy" in config.LABEL_MAP.values()


def test_seed_is_int():
    assert isinstance(config.SEED, int)


def test_hyperparameters_are_positive():
    assert config.LR > 0
    assert config.MIN_LR > 0
    assert config.BATCH_SIZE > 0
    assert config.EPOCHS > 0
    assert config.WEIGHT_DECAY >= 0


def test_scheduler_is_valid():
    valid_schedulers = {"ReduceLROnPlateau", "CosineAnnealingLR", "CosineAnnealingWarmRestarts"}
    assert config.SCHEDULER in valid_schedulers


def test_criterion_is_valid():
    valid_criteria = {"CrossEntropyLoss", "FocalCosineLoss", "BiTemperedLoss"}
    assert config.CRITERION in valid_criteria


def test_n_fold_greater_than_trn_fold():
    for fold in config.TRN_FOLD:
        assert fold < config.N_FOLD
