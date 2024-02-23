import logging
import yaml
# from fastapi import Depends, FastAPI, HTTPException, status
# import uvicorn
import mlflow
from pydantic import BaseModel
from service.data_prep import *
# import tvs_lce_app
logging.basicConfig(level=logging.INFO)

# load config
with open("service/config.yaml", "r") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
        print('APIKEY', CONFIG['APIKEY'])
    except yaml.YAMLError as exc:
        logging.error('Exception occurred while loading config.yaml', exc)

# load model as a PyFuncModel.
LOADED_MODEL = mlflow.pyfunc.load_model("model")

# fetching the input schema dictionary of the Loaded Model
input_schema_dict = LOADED_MODEL.metadata.get_input_schema().to_dict()

# fetching the list of input features
INPUT_FEATURES = get_list_of_input_variables_from_input_schema(input_schema_dict)
print("input features",INPUT_FEATURES)

# fetching the data type dictionary
data_type_dict = get_data_type_dict(input_schema_dict)


description = """
This API helps you to classify a Head Office (HO) lead (into HOT, WARM, COLD) based on the propensity to retail in real-time 🚀

#### Sample Request Body
```yaml
{
  "api_key":"0526817fcef64110938ef2c19126bc17",
  "enq_datetime": "2022-03-15T16:06:29.364040",
  "enq_source": "WEBSITE",
  "model_id": "10005000000600",
  "part_id": "N71900105D",
  "prospect_name": "Naga",
  "prospect_mobile_no": "9493965641",
  "prospect_pincode": "560099",
  "dealer_pincode": "560078",
  "buy_date_range": "with in 2 days",
  "int_finance": 1
}
```

#### Response
```yaml
{
    "retail_proba": "0.263338254775887",
    "predicted_label": "HOT"
}
```
"""

json={
  "api_key":"0526817fcef64110938ef2c19126bc17",
  "enq_datetime": "2022-03-15T16:06:29.364040",
  "enq_source": "WEBSITE",
  "model_id": "10005000000600",
  "part_id": "N71900105D",
  "prospect_name": "Naga",
  "prospect_mobile_no": "9493965641",
  "prospect_pincode": "560099",
  "dealer_pincode": "560078",
  "buy_date_range": "with in 2 days",
  "int_finance": 1
}

# create app instance
# app = FastAPI(title="Lead Classification Engine - Digital HO Model API",
#               description=description,
#               version="v0.1-beta")


# @app.get('/')
# def get_root():
#     return {'message': 'Welcome to the LCE Digital HO Model API'}


# class LCERequestBody(BaseModel):
#     api_key: str
#     enq_datetime: str
#     enq_source: str
#     model_id: str
#     part_id: str
#     prospect_name: str
#     prospect_mobile_no: str
#     prospect_pincode: str
#     dealer_pincode: str
#     buy_date_range: str
#     int_finance: int


def assign_label(pred_proba):
    if pred_proba >= 0.0647:
        return 'HOT'
    elif 0.0229 <= pred_proba < 0.0647:
        return 'WARM'
    return 'COLD'


# @app.post("/api/lce/ho")
def classify_lead(lce_data):
    print("Prediction has started")
    # print("Request Body", lce_data)
    data_to_log = lce_data
    # print(lce_data)
    # print(type(lce_data))
    # print(lce_data.get('api_key'))
    # print(type(lce_data.get('api_key')))
    # print(CONFIG['APIKEY'])
    # print(type(CONFIG['APIKEY']))
    # deleting the api key to avoid the logging of the api key value
    # del data_to_log['api_key']
    # if (lce_data.get('api_key') == CONFIG['APIKEY']):
    #     print('inside if condition')
    response = {"retail_proba": None,
                "predicted_label": "HOT"}
    # try:

    print('inside try')
    enq_info = {'finance_ho': lce_data.get('int_finance'), 'buy_date_range': lce_data.get('buy_date_range'),
                'enq_mode': get_source(lce_data.get('enq_source')),
                'proximity': get_proximity(lce_data.get('prospect_pincode'), lce_data.get('dealer_pincode')),
                'avg_spent_by_pincode': lookup_avg_spent_at_pincode(lce_data.get('prospect_pincode')),
                'len_name' : len(lce_data.get('prospect_name'))
                }
    # parse enq datetime
    enq_datetime = datetime.strptime(lce_data.get('enq_datetime'), "%Y-%m-%dT%H:%M:%S.%f")
    # extract datetime features
    enq_info.update(get_datetime_features(enq_datetime))
    # look up enquired vehicle info
    enq_info.update(get_vehicle_info(lce_data.get('part_id')))
    # pull enq counts
    enq_info.update(get_enq_counts(lce_data.get('prospect_mobile_no')))
    # look up geo information such as state, city
    enq_info.update(get_geo_info(lce_data.get('prospect_pincode')))
    # look up ex-showroom price
    enq_info['ex_showroom_price'] = get_ex_showroom_price(model=enq_info['enq_veh_model'],
                                                            variant='enq_veh_variant',
                                                            state='dms_cust_state',
                                                            year_month='year_month')

    dealer_attribute_list = ["dealership_name", "city_name", "dms_dlr_territory", "dlr_area_name", "zone_name",
                                "dms_dlr_pincode", "dlr_lat", "dlr_long"]
    dealer_attributes =dict(zip(dealer_attribute_list, [None]*len(dealer_attribute_list)))
    # dealer_attributes = get_dealer_attributes(lce_data.dealer_id)
    enq_info.update(dealer_attributes)

    enq_info['dms_cust_pincode'] = lce_data.get('prospect_pincode')
    enq_info['proximity'] = get_proximity(lce_data.get('prospect_pincode'), lce_data.get('dealer_pincode'))
    enq_info['proximity_seg'] = segment_proximity(get_proximity(lce_data.get('prospect_pincode'), lce_data.get('dealer_pincode')))
    # TODO: hard-coding this value as it's not available at the time of enquiry
    enq_info['followed_up_under'] = "Not Avaialble"

    enq_info['dlr_pi_state'] = None
    enq_info['dms_dlr_town'] = None
    enq_info['town_class'] = None
    enq_info['monthly_dlr_enq_vol'] = None
    enq_info['dlr_cnt']=None
    enq_info['Dealer_Type'] = None

    enq_df = pd.DataFrame.from_dict(enq_info, orient='index').T

    # Function to make sure that all the input features have the required data type
    enq_df = match_data_type_of_input_features(enq_df, data_type_dict)

    # enq_df.proximity = enq_df.proximity.astype(float)
    # enq_df.top_speed = enq_df.top_speed.astype(float)
    # enq_df.enq_count_utm_src = enq_df.enq_count_utm_src.astype(float)
    # enq_df.enq_count = enq_df.enq_count.astype(float)
    # enq_df.ex_showroom_price = enq_df.ex_showroom_price.astype(float)
    # enq_df.finance_ho = enq_df.finance_ho.astype('int32')
    # enq_df.enq_hour = enq_df.enq_hour.astype(float)

    feat_dict = enq_df[INPUT_FEATURES].to_dict(orient='index')

    index_ = list(feat_dict.keys())[0]
    data_to_log.update(feat_dict[index_])

    prob_to_retail = LOADED_MODEL.predict(enq_df[INPUT_FEATURES])[0]
    logging.info("Prediction has been Provided")

    response = {"retail_proba": str(prob_to_retail),
                "predicted_label": assign_label(prob_to_retail)}

    # print(response)

    # except Exception as e:
    #     logging.error('Exception occurred while predicting the label', e)


    data_to_log.update(response)

    # print(data_to_log)
    # print(response)
    # logging.critical(f'{data_to_log}')
    return response
    
    # raise HTTPException(
    #     status_code=status.HTTP_401_UNAUTHORIZED,
    #     detail="Invalid API Key",
    #     headers={"WWW-Authenticate": "Basic"},
    # )


if __name__ == "__main__":
#     # import uuid
#     # print(uuid.uuid4().hex)
#     # uvicorn.run(app)
    classify_lead(json)
