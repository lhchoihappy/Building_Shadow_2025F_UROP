# Heat Index Estimation (2025 Fall HKUST UROP Project)
- _Supervised by Prof. FUNG, Jimmy Chi Hung, and co-supervised by Dr. CHAN, Jimmy Wai Man_
- This program may have some bugs due to the current investigation and the progress of working.

<!-- ## Features: 
- Develop a hourly spatial map integrating the sunshine minutes of the building shadow and the solar irradiance in different points by using some radiation and building-shadow modules
- Perform various GIS works, like handling building shadows of different building morphologies, solar irradiance, and the extraction of the HK map -->

## How to use this repository

### Changing the file directory
In `main.py`, there are 2 major file input that you need to change when you use this program: 
1. In line 16, there is a `hk_building_path`. 
   - This is the path directory of the whole HK building Geojson file that allows you to choose the region you want to test for the building shadow analysis
2. In line 17, there is a `building_height_path` that is used to store the height data of the building 
   - In case the extracted Geojson file has very little building height value, like this Kennedy Town domain example, this program can automatically insert the height data into the extracted Geojson file for the further processing
   - These height value are manually added from the Open3Dhk

### Global Variables 
In `main.py`, there are quite a lot of global variables that you can feel free to change:
1. `lat1, lon1`, `lat2, lon2`: they are the boundary of the extracted map. You can change them for a different domain visualization
2. `date`: the date that you want to do the visualization (I have done a test on using differnt date, and it can successfully show different output than the current 2025-08-20 date)
3. `max_T, min_T`: the maximum temperature and the minimum temperature on the date specified based on the nearest HKO temperature station
4. `suntime`: the hour that you want to do the visualization
   
## Test Cases
I have selected several test cases within the Kennedy Town region with different building shapes and height for the test case testing. You can feel free to try each of these test cases. 
## Functions

## My weekly Progress: 
