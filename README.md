# CarbonMix

A time-series forecasting framework for long-term prediction on carbon flux and weather data.

## Project Structure

- **run_carbon_data.py** — Run experiments on carbon flux (carbon sink) data
- **run_weather_data.py** — Run experiments on weather data
- **run.py** — Core experiment entry point

## Usage

### Carbon Flux Data

```bash
python run_carbon_data.py
```

Data directory: `./dataset/carbon/`. Target variable: NEE (Net Ecosystem Exchange).

### Weather Data

```bash
python run_weather_data.py
```

Data directory: `./dataset/weather/`. Data file: `weather.csv`.

## Carbon Sink Dataset

**Copyright notice:** The carbon sink data is subject to copyright. Users must obtain the data themselves. We provide download links and a processing script for reference only.
- [Dataset 1 (SciDB)](https://www.scidb.cn/detail?dataSetId=824941006418870272&version=V1)
- [Dataset 2 (NESDC)](https://www.nesdc.org.cn/sdo/detail?id=64e6c4f07e2817429fbc7afa)
- [Dataset 3 (SciDB)](https://www.scidb.cn/detail?dataSetId=720626422036561920&version=V1)
- [Dataset 4 (SciDB)](https://www.scidb.cn/detail?dataSetId=4935daa458a34c3dae22a36cb317826c&version=V1)
- [Dataset 5 (SciDB File)](https://www.scidb.cn/file?fid=60504df8124e3600e55445d5&mode=front)
- [Dataset 6 (SciDB)](https://www.scidb.cn/detail?dataSetId=9b649cdd9cb143cc9b3188d7a6a38a31&version=V3)
- [Dataset 7 (NESDC)](https://www.nesdc.org.cn/sdo/detail?id=64e6c14f7e2817429fbc7af7)
- [Dataset 8 (SciDB)](https://www.scidb.cn/detail?dataSetId=755472332243337216&version=V2)
- [Dataset 9 (SciDB)](https://www.scidb.cn/detail?dataSetId=be0acc7ca1804710b363fab019ce8336&version=V4)
- [Dataset 10 (SciDB)](https://www.scidb.cn/detail?dataSetId=c800dd446426478abba3b6ec24757ade&version=V1)

**Data sources:** [www.scidb.cn](https://www.scidb.cn), [www.nesdc.org.cn](https://www.nesdc.org.cn). Register or log in on the dataset pages, then download according to the site instructions.

### Processing

A processing script is provided to convert the raw data into the CSV format expected by this project. Place the processed files under `./dataset/carbon/`.

