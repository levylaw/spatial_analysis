# -*- coding: utf-8 -*-
import numpy as np
import geopandas
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity


def create_train_X(pop_point_4326:geopandas.GeoDataFrame, pop_field:str):
    '''
    Parameters
    ----------
    pop_point_4326 : geopandas.GeoDataFrame
        population GeoDataFrame with epsg:4326 crs.
    pop_field : str
        names of the columns which contain the population number.

    Returns
    -------
    train_X : TYPE
        training set of the pop_kde function.
    sample_weight : TYPE
        weight parameters of the pop_kde function.

    '''
    pop_point_32651 = pop_point_4326.to_crs({'init':'epsg:32651'})
    
    train_X = []
    sample_weight = np.array([])
    for index,i in pop_point_32651.iterrows():
        lng_lat_list = np.array([])
        lng_lat_list = np.append(lng_lat_list,i.geometry.x)
        lng_lat_list = np.append(lng_lat_list,i.geometry.y)
        train_X.append(lng_lat_list)
        sample_weight = np.append(sample_weight, i[pop_field])

    sample_weight = np.nan_to_num(sample_weight)
    train_X = np.array(train_X)
    print('run pop_density.create_train_X successfully\n')
    return train_X, sample_weight

def construct_grids(spatial_range:tuple, row_col_shape:tuple):
    """
    the form of spatial_range tuple accords with the return value of the gdal founction GetGeoTransform()
    In a north up image, spatial_range[1] is the pixel width
    and spatial_range[5] is the pixel height, which is a negative value
    The upper left corner of the upper left pixel is at position (spatial_range[0],spatial_range[3])
    """
    
    pixel_width = spatial_range[1]
    pixel_height = spatial_range[5]
    x_min = spatial_range[0]
    x_max = x_min + row_col_shape[1] * pixel_width
    y_max = spatial_range[3]
    y_min = y_max + row_col_shape[0] * pixel_height
    
    x_grid = np.arange(x_min, x_max, pixel_width) + (pixel_width/2)
    y_grid = np.arange(y_max, y_min, pixel_height) + (pixel_height/2)
    
    X, Y = np.meshgrid(x_grid, y_grid)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    print('run pop_density.construct_grids successfully\n')
    
    return xy

def pop_kde(spatial_range:tuple, row_col_shape:tuple, band_width:int, 
            xy:np.ndarray, train_X:np.ndarray, sample_weight:np.ndarray):
    
    kde = KernelDensity(bandwidth= band_width, metric='euclidean', kernel='gaussian')
    kde.fit(train_X, sample_weight)
    
    #人口的概率密度分布
    population_prob_density =  np.exp(kde.score_samples(xy))
    #像元中人口分布的概率
    population_prob_in_pixel = population_prob_density *  spatial_range[1] * np.abs(spatial_range[5]) 
    #像元中人口分布的数量
    population_in_pixel = np.round(population_prob_in_pixel * sample_weight.sum(),0)

    population_in_pixel_32651 = geopandas.GeoDataFrame(population_in_pixel,crs='+init=epsg:32651')
    point_list = []
    for i in xy:
        point = Point(i[0], i[1])
        point_list.append(point)

    population_in_pixel_32651.geometry = point_list
    population_in_pixel_32651.columns = ['population','geometry']
    population_in_pixel_32651 = population_in_pixel_32651.drop(population_in_pixel_32651[population_in_pixel_32651['population']==0].index)
    population_in_pixel_4326 = population_in_pixel_32651.to_crs({'init':'epsg:4326'})
    
    print('run pop_density.pop_kde successfully\n')
    return population_in_pixel_4326
    
if __name__ == '__main__':
    import os
    #file_path = 'E:\\suzhou_gongan\\suzhou_data\\人口数据\\yidong_pop_jzgz_191218.geojson'
    #out_path =  'E:\\suzhou_gongan\\suzhou_data\\geojson\\'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    SOURCE_DATA_DIR = os.path.join(BASE_DIR, 'source_data')
    
    pop_gdf = geopandas.read_file('E:\\suzhou_gongan\\suzhou_data\\人口数据\\yidong_pop_jzgz_191218.geojson',driver='GeoJSON', encoding='UTF-8')
    '''
    pop_gdf = geopandas.read_file(
        os.path.join(SOURCE_DATA_DIR, 'yidong_pop_jzgz_191218.geojson'),
        driver='GeoJSON', encoding='UTF-8')
    '''
    #基站点对应的栅格范围，原始点位坐标系为wgs84
    spatial_range = (215816.8993, 1000.0, 0.0, 3547245.4869, 0.0, -1000.0)
    #栅格像元数
    row_col_shape = (144, 127)
    band_width = 2500
    population_in_pixel_list = []
    for pop_field in ['cnt_user_juzhu','cnt_user_gongzuo']:
        train_X, sample_weight = create_train_X(pop_gdf, pop_field)
        xy = construct_grids(spatial_range, row_col_shape)
        population_in_pixel_list.append(pop_kde(spatial_range, row_col_shape, band_width, xy, train_X, sample_weight))
        
    pop_juzhu_in_pixel, pop_gongzuo_in_pixel = population_in_pixel_list
    print(pop_juzhu_in_pixel.head(10))
