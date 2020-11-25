# OutlierHub

a curated hub for outlier and anomaly datasets built on top of [datastack](https://github.com/le1nux/datastack).

Currentyly supported datasets:

* ATIS
* ARRHYTHMIA
* KDD
* REUTERS
* TREC

# Install 

Clone or download the repository and `cd` into OutlierHub's root folder and run

```bash
pip install src/
```

# Usage

Each dataset has a factory that can be instantiated with a `StorageConnector` providing the IO operations. 
After factory instantiation, the factory is able to create a dataset iterator when calling its member method `get_dataset_iterator(split="<your split name>")`  

**Copyright 2020 Fraunhofer IAIS**

