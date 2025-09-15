from datasets import load_dataset, DatasetDict
ds_any = load_dataset("orcn/predictions", name="default", streaming=False)
d = ds_any['train'] if isinstance(ds_any, DatasetDict) and 'train' in ds_any else (ds_any[next(iter(ds_any.keys()))] if isinstance(ds_any, DatasetDict) else ds_any)
print("columns:", d.column_names, "rows:", len(d))
if len(d):
    ex = d[0]
    print("has images:", ('images' in ex) and (ex['images'] is not None))
    pred = ex.get('predictions')
    print("pred type:", type(pred), "preview:", (pred[:200] if isinstance(pred, str) else pred))
