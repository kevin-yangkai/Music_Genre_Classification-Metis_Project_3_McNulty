import csv
import pandas as pd
import pickle
import re
import googlemaps

# this function will grab a list of links to grab dataframes from
def open_links_from_CVS(file_name):
    with open(file_name, newline='', encoding='utf-8') as f:
        file = csv.reader(f, delimiter=',')
        # file = csv.reader(f)
        list_of_string = []
        for current in file:
            for more_current in current:
                list_of_string.append(more_current)
    return list_of_string


# this function will concatinate all dataframes pulled from web from a list of strings containing the links
def concat_data_frames_from_web(list_of_links):
    dframes_to_concat = []
    for link in list_of_links:
        dframes_to_concat.append(pd.read_csv(link))
    concated_df = pd.concat(dframes_to_concat)
    return concated_df


# this function will save a dataframe as a pickle
def save_dataframe_as_pickle(frame_to_save, save_name):
    with open(save_name, 'wb') as f:
        pickle.dump(frame_to_save, f)


# this function will open a dataframe as a pickle
def open_dataframe_pickle(name_of_pickle):
    with open(name_of_pickle, 'rb') as f:
        df_from_pickle = pickle.load(f)
    return df_from_pickle


# this function is for cleaning the station name data from typos in the dataset
def clean_station_names(dataframe_to_clean):
    dataframe_to_clean['STATION'] = [re.sub(r'AV', 'AVE', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'AVE{2}', 'AVE', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'RD', 'ROAD', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'PK', 'PARK', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'PARKWAY', 'PKWY', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'STS', 'ST', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'RACETR', 'TRACETRACK', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'PARKWY', 'PARKWAY', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'HWY', 'HIGHWAY', spot) for spot in dataframe_to_clean['STATION']]
    dataframe_to_clean['STATION'] = [re.sub(r'BLVD', 'BL', spot) for spot in dataframe_to_clean['STATION']]
    return dataframe_to_clean


# this function scrapes the MTA website and grabs all relevent links
def get_links(website_string):
    flag = 0
    instance_loc = 0
    links = []
    while flag != -1:
        instance_loc = website_string.find('<a href="data', instance_loc+1)
        if instance_loc == -1:
            flag = instance_loc
        else:
            ref_start = instance_loc+14
            ref_end = website_string.find('"', ref_start+1)
            links.append('http://web.mta.info/developers/data/' + website_string[ref_start:ref_end])
            flag = instance_loc
    return links


# this function saves scraped websites into a csv file
def save_links_to_CVS(links, file_name):
    with open(file_name, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(links)
    return


# this function will sort a list of strings and only keep those from a sepecified period of time
def sort_links_by_time_period(links):
    # m = re.search('\d\d[0][3-6]\d\d', current)
    focused_links = []
    for current in links:
        if re.search('[1][5-6][0][3-6]\d\d', current):
            focused_links.append(current)
    return focused_links


# this function will take an object and output a list of all its callable methods
def get_list_of_methods(object_to_check_out):
    methods_list = [method for method in dir(object_to_check_out) if callable(getattr(object_to_check_out, method))]
    return methods_list

#This function is inputed a list of locations and then returns three lists containined the latitude, longitude, and
# formal, properly formatted address for the location
def get_gps_coords(location_list):
    geocode_result = []
    lat = []
    lng = []
    formatted_address =[]
    gmaps = googlemaps.Client(key="AIzaSyBpK51cwVe1yBYAB4KWomfM3lx-QJSvklc")
    for current in location_list:
        temp = gmaps.geocode(current)
        try:
            geocode_result.append(temp[0])
        except:
            geocode_result.append(0)
    for current in geocode_result:
        if current == 0:
            lat.append(0)
            lng.append(0)
            formatted_address.append(0)
        else:
            lat.append(current['geometry']['location']['lat'])
            lng.append(current['geometry']['location']['lng'])
            formatted_address.append(current['formatted_address'])
    return lat, lng, formatted_address


# this function receives lists containing the lat, lng, and labels of points to place on a google map along with various
# styling parameters; will only plot the various locations with a single stype of marker
def google_map_builder_with_loc_markers(mycenter_lat, mycenter_lng, myzoom, myformat, mysize_x, mysize_y, mymaptype, mykey, mymarkersize, mylats, mylngs, label_color, label_name):
    root = 'http://maps.google.com/maps/api/staticmap?'
    label_name = [re.sub(r'[ ]', '+', current) for current in label_name]
    mylink = (root +
              'center=' + mycenter_lat + ',' + mycenter_lng +
              '&zoom=' + myzoom +
              '&size=' + mysize_x + 'x' + mysize_y +
              '&format=' + myformat +
              '&maptype=' + mymaptype)
    temp = ''
    for i in range(len(mylats)):
        temp = temp + '&markers=size:' + mymarkersize + '|' + 'color:' + label_color[i] + '|label:' + label_name[i] + '|' + str(mylats[i]) + ',' + str(mylngs[i])
    mylink = mylink + temp + '&key=' + mykey
    return mylink


def google_map_builder_two_marker_types(mycenter_lat, mycenter_lng, myzoom, myformat, mysize_x, mysize_y, mymaptype, mykey, mymarkersize, mylats_1, mylngs_1, label_color_1, label_name_1, mylats_2, mylngs_2, label_color_2, label_name_2):
    root = 'http://maps.google.com/maps/api/staticmap?'
    label_name_1 = [re.sub(r'[ ]', '+', current) for current in label_name_1]
    label_name_2 = [re.sub(r'[ ]', '+', current) for current in label_name_2]
    mylink = (root +
              'center=' + mycenter_lat + ',' + mycenter_lng +
              '&zoom=' + myzoom +
              '&size=' + mysize_x + 'x' + mysize_y +
              '&format=' + myformat +
              '&maptype=' + mymaptype)
    temp = ''
    for i in range(len(mylats_1)):
        temp = temp + '&markers=size:' + mymarkersize + '|' + 'color:' + label_color_1 + '|label:' + label_name_1[i] + '|' + str(mylats_1[i]) + ',' + str(mylngs_1[i])
    for i in range(len(mylats_2)):
        temp = temp + '&markers=size:' + mymarkersize + '|' + 'color:' + label_color_2 + '|' + str(mylats_2[i]) + ',' + str(mylngs_2[i])
    mylink = mylink + temp + '&key=' + mykey
    return mylink