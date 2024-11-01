# Contributing

## Add new mapping methods

WIP.

## Add new generation methods

All new generation methods should be a subclass of `map2text.model.base.IdeaGenerator`. See `map2text.model.non_trainable_gen.PlagiarismGenerator` for an example.

If you are adding a new method that requires training, you may put train and inference code separately. Check `map2text.model.trainable_ffn` for an example.

Here is a checklist for adding a new generation method:

- [ ] Implement the method in anew class in `map2text.model`
- [ ] Add the config file in `map2text.configs`
- [ ] Add the import in `map2text.quick_start.GenerationExperiment.from_config` method.

After that, you can use the new method by `python map2text/quick_start.py map2text/configs/<dataset>/<method_name>.yaml`.
