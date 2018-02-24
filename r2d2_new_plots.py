import ast
import json
import warnings

from collections import OrderedDict
from datetime import datetime, timedelta
from sqlalchemy import create_engine

import boto
import findspark
import numpy as np
import pandas as pd

findspark.init()

from bokeh.embed import components
from bokeh.plotting import *
from bokeh.models import HoverTool, Title
from flask import Flask, render_template, redirect, url_for, request
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import Form
from pyspark import SparkContext, HiveContext
from site_data import get_all_site_data
from wtforms import SubmitField, SelectField, StringField, SelectMultipleField, widgets
from wtforms.fields.html5 import IntegerRangeField
from wtforms_html5 import AutoAttrMeta
from wtforms.validators import DataRequired, NumberRange, ValidationError


app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

manager = Manager(app)
bootstrap = Bootstrap(app)
moment = Moment(app)

env_config = json.load(open('/etc/side_config.json', 'r'))

METRICS = ''
MESSAGE = ''
_mc = []
global site_df


ACCOUNT_QUERY = '''SELECT
                      d.full_date AS Date,
                      SUM(a.adwords_revenue) AS revenue,
                      SUM(COALESCE(a.adwords_orders, a.adwords_conversions, 0)) AS orders,
                      SUM(a.adwords_clicks) AS clicks,
                      SUM(a.adwords_impressions) AS impressions,
                      SUM(a.adwords_cost) AS cost
                    FROM
                      f_product_adgroup_performance a
                    JOIN
                      d_date d
                    ON
                      a.d_date_id = d.id
                    WHERE
                      d.full_date >= '{full_date}'
                    GROUP BY
                      d.full_date'''

SITE_DATA_QUERY = '''SELECT
                       d.full_date AS Date,
                       p.orders AS site_orders,
                       p.revenue AS site_revenue
                     FROM
                       f_site_performance p
                     JOIN
                       d_date d
                     ON
                       p.d_date_id = d.id
                     WHERE
                       d.full_date >= '{full_date}'
                     GROUP BY
                       d.full_date'''

PERF_PRED_QUERY = '''SELECT
                       site_id,
                       prediction_date
                     FROM
                       prediction_performance
                     WHERE
                       prediction_date >= '{date}'
                     GROUP BY
                       site_id,
                       prediction_date
                    '''


def lookback_validator(form, field):
    if int(field.data) < 14:
        raise ValidationError('The lookback must be longer than 14 days')


class MultiCheckboxField(SelectMultipleField):
    """
    A multiple-select, except displays a list of checkboxes.

    Iterating the field will produce subfields, allowing custom rendering of
    the enclosed checkbox fields.
    """
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class MainForm(Form):
    rank_models = SubmitField('Rank All Models')
    compare_models = SubmitField('Compare Models')
    see_importance = SubmitField('Variable Importance')
    daily_changes = SubmitField('Daily Highlights')
    adjust_threshold = SubmitField('Adjust Thresholds')
    run_simulation = SubmitField('Simulation Bake Off')
    channel_performance = SubmitField('Channel Performance')
    client_checkbox = SubmitField('Client Checkbox')
    prediction_performance = SubmitField('Prediction Performance')


class DropForm(Form):
    model_list = SelectField('Available Models', validators=[DataRequired()])
    sub = SubmitField("Query Variable Importance")


class FirstDrop(Form):
    client_list = SelectField('Select Client',  validators=[DataRequired()])
    run_queries = SubmitField('Get Available Dates')


class ClientSlider(Form):
    client_list = SelectField('Select Client',  validators=[DataRequired()])
    run_queries = SubmitField('View Choices')


class SecondDrop(Form):
    date_list = SelectField('Specify Date', validators=[DataRequired()])
    run_queries = SubmitField('Gather Data')


class ProductDisplay(Form):
    product_list = SelectField('Select Ranked Product', validators=[DataRequired()])
    next_prod = SubmitField('Show Next Product')


class ComparisonForm(Form):
    client_list_a = SelectField('Select Client/General A', validators=[DataRequired()])
    type_list_a = SelectField('Select Model Type A', validators=[DataRequired()])
    platform_list_a = SelectField('Select Platform A', validators=[DataRequired()])

    client_list_b = SelectField('Select Client/General B', validators=[DataRequired()])
    type_list_b = SelectField('Select Model Type B', validators=[DataRequired()])
    platform_list_b = SelectField('Select Platform B', validators=[DataRequired()])

    run_reports = SubmitField('Run Reports')
    rank_left = SubmitField('Rank by Left')
    rank_right = SubmitField('Rank by Right')


class Sliders(Form):
    class Meta(AutoAttrMeta):
        pass
    slider = IntegerRangeField('CM Threshold', validators=[NumberRange(min=0, max=4)], default=2)
    check_changes = SubmitField('Test Threshold for Bid Changes')


class ChannelMetrics(Form):
    client_list = SelectField('Specify Client', validators=[DataRequired()])
    metrics_list = SelectField('Specify Metric', validators=[DataRequired()])
    lookback = StringField('Lookback', validators=[DataRequired(), lookback_validator])
    generate_graph = SubmitField('Graph Metrics')


class PredictionPerformance(Form):
    client_list = SelectField('Specify Client', validators=[DataRequired()])
    date_list = SelectField('Select Prediction Date', validators=[DataRequired()])
    gen_pred = SubmitField('Evaluate Performance')


class ClientEntry(Form):
    run_reports = SubmitField('Run Reports', validators=[DataRequired()])
    client_chooser = MultiCheckboxField("Clients")


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


def prep_control_chart_data(_aq, _sq, _stats):
    account_data = pd.read_sql(_aq, _stats)
    site_data = pd.read_sql(_sq, _stats)
    jd = account_data.merge(site_data, on='Date')
    jd['pct_site_orders'] = jd['orders']/jd['site_orders']
    jd['pct_site_revenue'] = jd['revenue']/jd['site_revenue']
    jd['aov'] = jd['revenue']/jd['orders']
    jd['cr'] = jd['orders']/jd['clicks']
    jd['ctr'] = jd['clicks']/jd['impressions']
    jd['cpc'] = jd['cost']/jd['clicks']
    jd.fillna(0, inplace=True)
    jd['orders_7_avg'] = jd['orders'].rolling(window=7, center=False).mean()
    jd['revenue_7_avg'] = jd['revenue'].rolling(window=7, center=False).mean()
    jd['clicks_7_avg'] = jd['clicks'].rolling(window=7, center=False).mean()
    jd['impressions_7_avg'] = jd['impressions'].rolling(window=7, center=False).mean()
    jd['cost_7_avg'] = jd['cost'].rolling(window=7, center=False).mean()

    jd['sum_orders'] = jd['orders'].rolling(window=7, center=False).sum()
    jd['sum_clicks'] = jd['clicks'].rolling(window=7, center=False).sum()
    jd['sum_impressions'] = jd['impressions'].rolling(window=7, center=False).sum()
    jd['sum_revenue'] = jd['revenue'].rolling(window=7, center=False).sum()
    jd['sum_cost'] = jd['cost'].rolling(window=7, center=False).sum()

    jd['aov_7_avg'] = jd['sum_revenue']/jd['sum_orders']
    jd['cr_7_avg'] = jd['sum_orders']/jd['sum_clicks']
    jd['ctr_7_avg'] = jd['sum_clicks']/jd['sum_impressions']
    jd['cpc_7_avg'] = jd['sum_cost']/jd['sum_clicks']

    jd['cos_m'] = jd['cost']/jd['revenue']
    jd['cos_m_7_avg'] = jd['sum_cost']/jd['sum_revenue']

    jd['roas'] = jd['revenue']/jd['cost']
    jd['roas_7_avg'] = jd['sum_revenue']/jd['sum_cost']

    jd.drop(['sum_orders', 'sum_clicks', 'sum_impressions', 'sum_revenue', 'sum_cost'], axis=1, inplace=True)

    jd = jd.dropna()
    jd.reset_index(drop=True, inplace=True)

    return jd


def detect_mean_runs(_df, _metric, _ou):
    if _ou == 'above':
        mean = _df[_metric].mean()
        _df['days_above_mean'] = _df[_metric] - mean
        _df['above_mean_flag'] = np.where(_df['days_above_mean'] > 0, 1, 0)
        _df['grouper'] = (_df['above_mean_flag']!=_df['above_mean_flag'].shift()).cumsum()
        _df['above_mean_runs'] = _df.groupby('grouper')['above_mean_flag'].cumsum()

    elif _ou == 'below':
        mean = _df[_metric].mean()
        _df['days_below_mean'] = _df[_metric] - mean
        _df['below_mean_flag'] = np.where(_df['days_below_mean'] < 0, 1, 0)
        _df['grouper'] = (_df['below_mean_flag']!=_df['below_mean_flag'].shift()).cumsum()
        _df['below_mean_runs'] = _df.groupby('grouper')['below_mean_flag'].cumsum()


def detect_slope_changes(_df, _metric, _ou):
    if _ou == 'negative':
        _df['yesterday'] = _df[_metric].shift()
        _df['delta'] = _df[_metric] - _df['yesterday']
        _df['negative_slope_flag'] = np.where(_df['delta'] < 0, 1, 0)
        _df['grouper'] = (_df['negative_slope_flag']!=_df['negative_slope_flag'].shift()).cumsum()
        _df['negative_slope_runs'] = _df.groupby('grouper')['negative_slope_flag'].cumsum()
    elif _ou == 'positive':
        _df['yesterday'] = _df[_metric].shift()
        _df['delta'] = _df[_metric] - _df['yesterday']
        _df['positive_slope_flag'] = np.where(_df['delta'] > 0, 1, 0)
        _df['grouper'] = (_df['positive_slope_flag']!=_df['positive_slope_flag'].shift()).cumsum()
        _df['positive_slope_runs'] = _df.groupby('grouper')['positive_slope_flag'].cumsum()


def detect_pixel_issues(_df):
    _df = _df.tail(n=7)
    no_rev_count = len(_df[_df['revenue'] == 0])
    return no_rev_count


def check_for_alerts(_dict, _client_name):
    alert_dict = {}
    for k in _dict.keys():
        all_alerts = list(set(_dict[k].values()))
        all_alerts.remove('09_no_alert')
        alerts_image_config = {
        '02_high_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="high_alert.png"), cl=_client_name, m=k),
        '03_medium_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="medium_alert.png"), cl=_client_name, m=k),
        '08_low_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="low_alert.png"), cl=_client_name, m=k),
        '09_no_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="no_alert.png"), cl=_client_name, m=k),
        '04_below_mean_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="under_average.jpg"), cl=_client_name, m=k),
        '06_above_mean_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="over_average.jpg"), cl=_client_name, m=k),
        '07_positive_slope_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="up_trend.png"), cl=_client_name, m=k),
        '05_negative_slope_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="down_trend.png"), cl=_client_name, m=k),
        '01_revenue_pixel_alert': '''<a href="http://emr.clarityinretail.com:5000/channel_metrics?client={cl}&metric={m}"><img src={url} style="width: 50%; height: 50%" align="middle"/>'''.format(
            url=url_for('static', filename="bad_pixel.jpg"), cl=_client_name, m=k)}
        if len(all_alerts) == 0:
            alert_dict[k] = alerts_image_config['09_no_alert']
        else:
            all_alerts.sort()
            alert_dict[k] = alerts_image_config[all_alerts[0]]

    return alert_dict


def compute_dashboard_summary(_health_df, _metrics_list, _N, _one_dev_thresh, _two_dev_thresh):

    results_dict = {}

    for _metric in _metrics_list:

        metric_dict = {}

        detect_mean_runs(_health_df, _metric, 'above')
        detect_mean_runs(_health_df, _metric, 'below')
        detect_slope_changes(_health_df, _metric, 'negative')
        detect_slope_changes(_health_df, _metric, 'positive')

        _m_me = _metric + '_mean'
        _m_sd = _metric + '_sd'
        _health_df[_m_me] = _health_df[_metric].mean()
        _health_df[_m_sd] = _health_df[_metric].std()
        metric_c1_upper = (_health_df[_m_me] + _health_df[_m_sd])[0]
        metric_c1_lower = (_health_df[_m_me] - _health_df[_m_sd])[0]

        cutback_df = _health_df.tail(_N)

        if _metric == 'revenue':
            no_rev_days = detect_pixel_issues(cutback_df)
            if no_rev_days > round(.5 * _N):
                metric_dict['tracking_alert'] = '01_revenue_pixel_alert'

        gt1 = len(cutback_df[cutback_df[_metric] > metric_c1_upper].index)
        lt1 = len(cutback_df[cutback_df[_metric] < metric_c1_lower].index)
        metric_c2_upper = (_health_df[_m_me] + (2*_health_df[_m_sd]))[0]
        metric_c2_lower = (_health_df[_m_me] - (2*_health_df[_m_sd]))[0]
        gt2 = len(cutback_df[cutback_df[_metric] > metric_c2_upper].index)
        lt2 = len(cutback_df[cutback_df[_metric] < metric_c2_lower].index)
        metric_c3_upper = (_health_df[_m_me] + (3*_health_df[_m_sd]))[0]
        metric_c3_lower = (_health_df[_m_me] - (3*_health_df[_m_sd]))[0]
        gt3 = len(cutback_df[cutback_df[_metric] > metric_c3_upper].index)
        lt3 = len(cutback_df[cutback_df[_metric] < metric_c3_lower].index)
        if ((gt3 > 0) | (lt3 > 0)):
            metric_dict['deviation'] = '02_high_alert'
        elif ((gt2 >= (round(_two_dev_thresh * _N))) | (lt2 >= (round(_two_dev_thresh * _N)))):
            metric_dict['deviation'] = '03_medium_alert'
        elif ((gt1 >= round(_one_dev_thresh * _N)) | (lt1 >= round(_one_dev_thresh * _N))):
            metric_dict['deviation'] = '08_low_alert'
        else:
            metric_dict['deviation'] = '09_no_alert'

        if cutback_df['below_mean_runs'].max() >= _N-1:
            metric_dict['below_mean_trend'] = '04_below_mean_alert'
        else:
            metric_dict['below_mean_trend'] = '09_no_alert'

        if cutback_df['above_mean_runs'].max() >= _N-1:
            metric_dict['above_mean_trend'] = '06_above_mean_alert'
        else:
            metric_dict['above_mean_trend'] = '09_no_alert'

        if (cutback_df['negative_slope_runs'].max() >= _N-1) & (cutback_df['positive_slope_runs'].max() < _N-1):
            metric_dict['negative_slope_trend'] = '05_negative_slope_alert'
        else:
            metric_dict['negative_slope_trend'] = '09_no_alert'

        if (cutback_df['positive_slope_runs'].max() >= _N-1) & (cutback_df['negative_slope_runs'].max() < _N-1):
            metric_dict['positive_slope_trend'] = '07_positive_slope_alert'
        else:
            metric_dict['positive_slope_trend'] = '09_no_alert'

        if (cutback_df['positive_slope_runs'].max() >= _N-1) & (cutback_df['negative_slope_runs'].max() >= _N-1):
            print("It's a wash, metric {m} has experienced slope volatility over the past {n} days".format(m=_metric,
                                                                                                          n=_N))
        else:
            pass

        results_dict[_metric] = metric_dict

    return results_dict


def run_analysis(_stats, _client_name):
    look_window = 60

    aq = ACCOUNT_QUERY.format(full_date=(datetime.now() -
                                                 timedelta(look_window)).strftime('%Y-%m-%d'))

    sq = SITE_DATA_QUERY.format(full_date=(datetime.now() -
                                                   timedelta(look_window)).strftime('%Y-%m-%d'))

    df = prep_control_chart_data(aq, sq, _stats)
    warnings.filterwarnings('ignore')

    client_health_metrics = ['revenue', 'cost', 'orders', 'clicks', 'impressions', 'roas']
    health_df = df[client_health_metrics]
    N = 10
    two_dev_thresh = .4
    one_dev_thresh = .6

    res_d = compute_dashboard_summary(health_df, client_health_metrics, N, one_dev_thresh, two_dev_thresh)
    ad = check_for_alerts(res_d, _client_name)
    return ad


def create_trended_plots(_df, _variable, _level, _sn):
    ''' A function for graphing control charts for CSM/Analysts
    _df: the input DataFrame
    _variable: the metric to be graphed
    _level: site or channel metrics
    '''
    variables_list = [x for x in _df.columns if _variable in x]
    site_variables = [x for x in variables_list if 'site' in x]
    channel_variables = list(set(variables_list) - set(site_variables))
    site_variables.append('Date')
    channel_variables.append('Date')
    channel_df = _df[sorted(channel_variables)]
    site_daf = _df[sorted(site_variables)]

    if _level == 'site':

        _metric = _variable
        TOOLS = "wheel_zoom,box_zoom,reset,hover,previewsave"

        x_ax = site_daf['Date'].tolist()
        precision = 2
        site_daf[_metric] = site_daf[_metric].round(decimals=precision)
        y_ax = site_daf[_metric].tolist()

        p = figure(plot_height=600,
                   plot_width=1100,
                   x_axis_type='datetime',
                   tools=TOOLS,
                   toolbar_location="above")

        p.title.text = '{cl} Google Shopping as a Pct of Site {var}'.format(cl=_sn.title(),
                                                                            var=_variable.title())
        p.title.text_font_size = '18px'
        p.title.align = 'center'
        p.line(x_ax, y_ax, color="black", line_width=3)
        p.circle(x_ax, y_ax, fill_color="black", size=8)

        p.add_layout(Title(text='Date', align='center'), 'below')
        p.add_layout(Title(text='Daily Percent of Site {var}'.format(var=_variable.title()),
                           align='center'), 'left')

        var_mean = site_daf[_metric].mean()
        var_sd = site_daf[_metric].std()

        mean_plt = [var_mean.round(decimals=precision)] * len(x_ax)
        plus_one_sd = mean_plt + (var_sd*1).round(decimals=precision)
        minus_one_sd = mean_plt - (var_sd*1).round(decimals=precision)
        plus_two_sd = mean_plt + (var_sd*2).round(decimals=precision)
        minus_two_sd = mean_plt - (var_sd*2).round(decimals=precision)
        plus_three_sd = mean_plt + (var_sd*3).round(decimals=precision)
        minus_three_sd = mean_plt - (var_sd*3).round(decimals=precision)

        p.line(x_ax, mean_plt, color='blue', line_width=1)
        p.line(x_ax, plus_one_sd, color='green', line_width=1)
        p.line(x_ax, minus_one_sd, color='green', line_width=1)
        p.line(x_ax, plus_two_sd, color='yellow', line_width=1)
        p.line(x_ax, minus_two_sd, color='yellow', line_width=1)
        p.line(x_ax, plus_three_sd, color='red', line_width=1)
        p.line(x_ax, minus_three_sd, color='red', line_width=1)

        hover = p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("Value", "@y")
            #("Date", "@x".strftime('%Y-%m-%d %H:%M:%S'))
        ])

        figjs, figdiv = components(p)

    elif _level == 'channel':

        x_ax = channel_df['Date'].tolist()
        _metric = _variable
        _metric_2 = _variable + '_7_avg'
        precision = 0
        channel_df[_metric] = channel_df[_metric].round(decimals=precision)
        channel_df[_metric_2] = channel_df[_metric_2].round(decimals=precision)
        y_ax = channel_df[_metric].tolist()
        y2_ax = channel_df[_metric_2].tolist()

        TOOLS = "wheel_zoom,box_zoom,reset,hover,previewsave"
        p = figure(plot_height=600,
                   plot_width=1100,
                   x_axis_type='datetime',
                   tools=TOOLS,
                   toolbar_location="above")

        p.title.text = '{cl} Google Shopping {var} Trended'.format(cl=_sn.title(),
                                                                   var=_metric.title())
        p.title.text_font_size = '18px'
        p.title.align = 'center'
        p.line(x_ax, y_ax, color="black", line_width=3)
        p.line(x_ax, y2_ax, color='orange', line_width=2)
        p.circle(x_ax, y_ax, fill_color="black", size=8)
        p.circle(x_ax, y2_ax, fill_color="orange", size=6)

        p.add_layout(Title(text='Date', align='center'), 'below')
        p.add_layout(Title(text='Google Shopping Daily {var}'.format(var=_metric.title()),
                           align='center'), 'left')

        var_mean = channel_df[_metric].mean()
        var_sd = channel_df[_metric].std()

        mean_plt = [var_mean.round(decimals=precision)] * len(x_ax)
        plus_one_sd = mean_plt + (var_sd*1).round(decimals=precision)
        minus_one_sd = mean_plt - (var_sd*1).round(decimals=precision)
        plus_two_sd = mean_plt + (var_sd*2).round(decimals=precision)
        minus_two_sd = mean_plt - (var_sd*2).round(decimals=precision)
        plus_three_sd = mean_plt + (var_sd*3).round(decimals=precision)
        minus_three_sd = mean_plt - (var_sd*3).round(decimals=precision)

        p.line(x_ax, mean_plt, color='blue', line_width=1)
        p.line(x_ax, plus_one_sd, color='green', line_width=1)
        p.line(x_ax, minus_one_sd, color='green', line_width=1)
        p.line(x_ax, plus_two_sd, color='yellow', line_width=1)
        p.line(x_ax, minus_two_sd, color='yellow', line_width=1)
        p.line(x_ax, plus_three_sd, color='red', line_width=1)
        p.line(x_ax, minus_three_sd, color='red', line_width=1)

        hover = p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("Value", "@y")
            #("Date", "@x".strftime('%Y-%m-%d %H:%M:%S'))
        ])

        figjs, figdiv = components(p)

    return figjs, figdiv


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/ranking_rmse', methods=['GET', 'POST'])
def ranking_rmse():
    _df = pd.read_csv('~/monitoring/model_eval_{d}.csv'.
                      format(d=datetime.today().strftime('%Y-%m-%d')))
    _df = _df.sort_values('rmse')
    _df.reset_index(inplace=True, drop=True)
    _df = _df[['model_id', 'short_name', 'type', 'rmse']]
    return render_template('results.html', tables=[_df.to_html()], message='Ranked by RMSE')


@app.route('/importance', methods=['GET', 'POST'])
def importance():
    _df = pd.read_csv('~/monitoring/model_eval_{d}.csv'.
                      format(d=datetime.today().strftime('%Y-%m-%d')))
    mc = _df['model'].tolist()
    form = DropForm()
    form.model_list.choices = [(i, i) for i in mc]
    if form.validate_on_submit():
        f = form.model_list.data
        blob = _df[_df['model'] == f]
        blob.sort_values('rmse', inplace=True)
        blob.reset_index(inplace=True, drop=True)
        blob = blob['vi'][0]
        v_imp = pd.DataFrame.from_dict(ast.literal_eval(blob), orient='index')
        v_imp.reset_index(drop=False, inplace=True)
        v_imp.columns = ['variables', 'importance']
        v_imp.sort_values('importance', inplace=True, ascending=False)
        v_imp.reset_index(drop=True, inplace=True)
        return render_template('vimp_results.html', tables=[v_imp.to_html()],
                               message='Importance for model {m}'.format(m=f), form=form)
    return render_template('base.html', form=form)


def get_comparison_choices():
    _df = pd.read_csv('~/monitoring/model_eval_{d}.csv'.
                      format(d=datetime.today().strftime('%Y-%m-%d')))
    client = list(set(_df['short_name'].tolist()))
    model_types = list(set(_df['type'].tolist()))
    platform = list(set(_df['ud'].tolist()))

    _d_r1 = pd.DataFrame.from_dict({0: 'Currently no results'}, orient='index').to_html()
    _d_r2 = pd.DataFrame.from_dict({0: 'Currently no results'}, orient='index').to_html()

    _form = ComparisonForm()
    _form.client_list_a.choices = [(i, i) for i in client]
    _form.client_list_b.choices = [(i, i) for i in client]

    _form.type_list_a.choices = [(i, i) for i in model_types]
    _form.type_list_b.choices = [(i, i) for i in model_types]

    _form.platform_list_a.choices = [(i, i) for i in platform]
    _form.platform_list_b.choices = [(i, i) for i in platform]

    return _form, _df, _d_r1, _d_r2


def generate_comparison_tables(_in_df, _c1, _m1, _p1, _c2, _m2, _p2, _dir):
        try:
            d_r1 = _in_df[(_in_df['short_name'] == _c1) & (_in_df['type'] == _m1) &
                          (_in_df['ud'] == _p1)]
            d_r1.reset_index(drop=True, inplace=True)
            blob1 = d_r1['vi'][0]
            v_imp1 = pd.DataFrame.from_dict(ast.literal_eval(blob1), orient='index')
            v_imp1.reset_index(drop=False, inplace=True)
            v_imp1.columns = ['variables', 'importance']
            v_imp1.sort_values('importance', inplace=True, ascending=False)
            v_imp1.reset_index(drop=True, inplace=True)
            mod_metrics1 = d_r1[['training', 'testing', 'rmse', 'start', 'end']].T
            mod_metrics1.columns = [str(d_r1['model'][0])]
            d_r1 = mod_metrics1.to_html()
        except:
            d_r1 = pd.DataFrame.from_dict({0: 'Invalid selection, '
                                              'please try again'}, orient='index').to_html()

        try:
            d_r2 = _in_df[(_in_df['short_name'] == _c2) & (_in_df['type'] == _m2) &
                          (_in_df['ud'] == _p2)]
            d_r2.reset_index(drop=True, inplace=True)
            blob2 = d_r2['vi'][0]
            v_imp2 = pd.DataFrame.from_dict(ast.literal_eval(blob2), orient='index')
            v_imp2.reset_index(drop=False, inplace=True)
            v_imp2.columns = ['variables', 'importance']
            v_imp2.sort_values('importance', inplace=True, ascending=False)
            v_imp2.reset_index(drop=True, inplace=True)
            mod_metrics2 = d_r2[['training', 'testing', 'rmse', 'start', 'end']].T
            mod_metrics2.columns = [str(d_r2['model'][0])]
            d_r2 = mod_metrics2.to_html()
        except:
            d_r2 = pd.DataFrame.from_dict({0: 'Invalid selection, '
                                              'please try again'}, orient='index').to_html()

        if _dir is 'left':
            v_imp1.sort_values('importance', inplace=True, ascending=False)
            v_imp2['variables'] = pd.Categorical(v_imp2['variables'], v_imp1['variables'].tolist())
            v_imp2 = v_imp2.sort('variables')
            vi1 = v_imp1.to_html()
            vi2 = v_imp2.to_html()
        elif _dir is 'right':
            v_imp2.sort_values('importance', inplace=True, ascending=False)
            v_imp1['variables'] = pd.Categorical(v_imp1['variables'], v_imp2['variables'].tolist())
            v_imp1 = v_imp1.sort('variables')
            vi1 = v_imp1.to_html()
            vi2 = v_imp2.to_html()
        else:
            vi1 = v_imp1.to_html()
            vi2 = v_imp2.to_html()

        return d_r1, d_r2, vi1, vi2


@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    global d_r1
    global d_r2
    form, df, d_r1, d_r2 = get_comparison_choices()
    if form.validate_on_submit() and form.run_reports.data:
        c1 = form.client_list_a.data
        m1 = form.type_list_a.data
        p1 = form.platform_list_a.data

        c2 = form.client_list_b.data
        m2 = form.type_list_b.data
        p2 = form.platform_list_b.data

        direct = 'None'

        d_r1, d_r2, vi1, vi2 = generate_comparison_tables(df, c1, m1, p1, c2, m2, p2, direct)
        return render_template('side_by_side.html', form=form,
                               tab1=d_r1, tab2=d_r2,
                               tab3=vi1, tab4=vi2)

    elif form.validate_on_submit() and form.rank_left.data:
        c1 = form.client_list_a.data
        m1 = form.type_list_a.data
        p1 = form.platform_list_a.data

        c2 = form.client_list_b.data
        m2 = form.type_list_b.data
        p2 = form.platform_list_b.data

        direct = 'left'

        d_r1, d_r2, vi1, vi2 = generate_comparison_tables(df, c1, m1, p1, c2, m2, p2, direct)
        return render_template('side_by_side.html', form=form,
                               tab1=d_r1, tab2=d_r2,
                               tab3=vi1, tab4=vi2)

    elif form.validate_on_submit() and form.rank_right.data:
        c1 = form.client_list_a.data
        m1 = form.type_list_a.data
        p1 = form.platform_list_a.data

        c2 = form.client_list_b.data
        m2 = form.type_list_b.data
        p2 = form.platform_list_b.data

        direct = 'right'

        d_r1, d_r2, vi1, vi2 = generate_comparison_tables(df, c1, m1, p1, c2, m2, p2, direct)
        return render_template('side_by_side.html', form=form,
                               tab1=d_r1, tab2=d_r2,
                               tab3=vi1, tab4=vi2)

    else:
        print('No clicks yet')

    return render_template('side_by_side.html', form=form,
                           tab1=d_r1, tab2=d_r2,
                           tab3=d_r1, tab4=d_r2)


@app.route('/daily_snapshot_clients', methods=['GET', 'POST'])
def daily_snapshot_clients():
    global client_choice
    conn = boto.connect_s3(host='s3.amazonaws.com')
    bucket = conn.get_bucket(env_config['s3_bucket'])
    folders = bucket.list('emr/hive/bid_changes/', '/')
    site_list = []
    for folder in folders:
        site_list.append(folder.name.replace('emr/hive/bid_changes/', '').strip('/'))

    site_list = [i for i in site_list if len(i) > 3]
    form = FirstDrop()
    form.client_list.choices = [(i, i) for i in site_list]

    if form.validate_on_submit():
        client_choice = form.client_list.data
        print(client_choice)
        return redirect(url_for('daily_snapshot_date'))

    return render_template('base.html', form=form)


@app.route('/daily_snapshot_date', methods=['GET', 'POST'])
def daily_snapshot_date():
    global site_df
    conn = boto.connect_s3(host='s3.amazonaws.com')
    bucket = conn.get_bucket(env_config['s3_bucket'])
    folders = bucket.list('emr/hive/bid_changes/{cli}/'.format(cli=client_choice), '/')
    avail_dates = []
    for folder in folders:
        avail_dates.append(folder.name.replace('emr/hive/bid_changes/{cli}/'.
                                               format(cli=client_choice), '').strip('/'))

    form = SecondDrop()
    form.date_list.choices = [(i, i) for i in avail_dates]

    if form.validate_on_submit():
        d1 = form.date_list.data
        key_name = 'emr/hive/bid_changes/{cli}/{date}/raw'.format(cli=client_choice,
                                                                                date=d1)
        conn = boto.connect_s3(host='s3.amazonaws.com')
        bucket = conn.get_bucket(env_config['s3_bucket'])
        df_list = []
        for p in bucket.list(prefix=key_name):
            print(p)
            try:
                df_list.append(pd.read_csv(p, sep=','))
            except:
                pass

        test_df = pd.concat(df_list)
        print(test_df.columns)
        test_df.sort_values('prediction', ascending=False, inplace=True)
        test_df.reset_index(inplace=False, drop=True)
        top5_df = test_df.head()
        # Make this guy global
        top5 = str(top5_df['product_id'].tolist()).strip('[]')
        config_filename = '/home/hadoop/data_flow/bid_rule/site_config.json'
        with open(config_filename, 'r') as config_file:
            config_contents = json.load(config_file)

        config = [c for c in config_contents if c['short_name'] == client_choice][0]
        site_id = config['site_id']
        get_sites = '''SELECT
                 product_id,
                 brand,
                 category,
                 get_json_object(decode(unbase64(catalog_json), 'UTF-8'), '$.price'),
                 get_json_object(decode(unbase64(catalog_json), 'UTF-8'), '$.cost'),
                 get_json_object(decode(unbase64(catalog_json), 'UTF-8'), '$.title'),
                 get_json_object(decode(unbase64(catalog_json), 'UTF-8'), '$.image_url')
               FROM
                 product_catalog
               WHERE
                 site_id = {sid}
               AND
                 product_id IN ({id_list})'''.format(sid=site_id, id_list=top5)

        try:
            sc = SparkContext(appName='R2D2_ProductCheck')
            hc = HiveContext(sc)
            site_ids = hc.sql(get_sites)
            site_df = site_ids.toPandas()
            site_df.columns = ['product_id', 'brand', 'category',
                               'price', 'cost', 'title', 'image_url']
            sc.stop()
        except (ValueError, AttributeError):
            #Need to find some way to kill the already running instance
            #sc.stop()
            sc = SparkContext(appName='R2D2_ProductCheck')
            hc = HiveContext(sc)
            site_ids = hc.sql(get_sites)
            site_df = site_ids.toPandas()
            site_df.columns = ['product_id', 'brand', 'category',
                               'price', 'cost', 'title', 'image_url']
            sc.stop()

        return redirect(url_for('product_image_display'))

    return render_template('base.html', form=form)


@app.route('/product_image_display', methods=['GET', 'POST'])
def product_image_display():
    global site_df
    # Need to read in the global top5
    form = ProductDisplay()
    form.product_list.choices = [(i, i+1) for i in range(len(site_df))]
    if form.product_list.data:
        if form.product_list.data != 'None':
            print(form.product_list.data)
            print(site_df['image_url'][int(form.product_list.data)])
            return render_template('display_products.html', form=form,
                                   img_link=site_df['image_url'][int(form.product_list.data)],
                                   prod_title=site_df['title'][int(form.product_list.data)],
                                   product_id=site_df['product_id'][int(form.product_list.data)],
                                   tables=[]) #Pass in the df here vi2 = v_imp2.to_html()
    return render_template('base.html', form=form)


@app.route('/threshold_sliders', methods=['GET', 'POST'])
def threshold_sliders():
    global client_choice
    conn = boto.connect_s3(host='s3.amazonaws.com')
    bucket = conn.get_bucket(env_config['s3_bucket'])
    folders = bucket.list('emr/hive/bid_changes/', '/')
    site_list = []
    for folder in folders:
        site_list.append(folder.name.replace('emr/hive/bid_changes/', '').strip('/'))

    site_list = [i for i in site_list if len(i) > 3]
    form = ClientSlider()
    form.client_list.choices = [(i, i) for i in site_list]

    if form.validate_on_submit():
        client_choice = form.client_list.data
        print(client_choice)
        return redirect(url_for('threshold_sliders2'))

    return render_template('base.html', form=form)


@app.route('/threshold_sliders2', methods=['GET', 'POST'])
def threshold_sliders2():
    global thresh_choices
    config_filename = '/home/hadoop/data_flow/bid_rule/site_config.json'
    with open(config_filename, 'r') as config_file:
        config_contents = json.load(config_file)
    config = [c for c in config_contents if c['short_name'] == client_choice][0]
    current_threshold = config['percent_prediction_threshold']
    form = Sliders()
    default_choices = [.7, .8, 1, 1.2, 1.3]
    thresh_choices = [i*current_threshold for i in default_choices]
    if form.validate_on_submit():
        d1 = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
        key_name = 'emr/hive/bid_changes/{cli}/{date}/raw'.format(cli=client_choice, date=d1)
        conn = boto.connect_s3(host='s3.amazonaws.com')
        bucket = conn.get_bucket(env_config['s3_bucket'])
        df_list = []
        for p in bucket.list(prefix=key_name):
            print(p)
            try:
                df_list.append(pd.read_csv(p, sep=','))
            except:
                pass

        increases_df = pd.concat(df_list)

        increases_df['predicted_cm_rank'] = increases_df['prediction'].rank(method='first', ascending=0)
        increases_df['predicted_cm_percentile'] = increases_df['predicted_cm_rank'] / len(increases_df)

        #increases_df = pd.read_csv('sample_increases.csv')
        increases_df = increases_df[increases_df['predicted_cm_percentile'] <
                                    thresh_choices[form.slider.data]]
        print('With a threshold of {n}, '
              'daily increases would have been {r}'.format(n=thresh_choices[form.slider.data],
                                                           r=len(increases_df)))
    return render_template('show_slider.html', form=form)


@app.route('/simulation', methods=['GET', 'POST'])
def simulation():
    return render_template('base.html', form=None)


@app.route('/channel_metrics', methods=['GET', 'POST'])
def channel_metrics():
    global info_df
    form = ChannelMetrics()
    client = request.args.get('client')
    metric = request.args.get('metric')
    lookback = 30
    params_exist = (client is not None and metric is not None)

    config = get_all_site_data()
    conf_info = []
    for blob in config:
        conf_info.append((blob['short_name'], blob['stats_db'], blob['pw']))
    info_df = pd.DataFrame(conf_info)
    info_df.columns = ['short_name', 'stats_db', 'pw']
    info_df.sort_values('short_name', inplace=True)
    client_list = info_df['short_name'].tolist()
    form.client_list.choices = [(i, i) for i in client_list]
    form.metrics_list.choices = [(i, i) for i in ['revenue', 'cost', 'clicks',
                                                  'orders', 'impressions', 'aov',
                                                  'cr', 'ctr', 'cos_m', 'cpc',
                                                  'pct_site_revenue',
                                                  'pct_site_orders']]
    if params_exist:
            form.lookback.default = lookback
            form.client_list.default = client
            form.metrics_list.default = metric

    if form.validate_on_submit() | params_exist:
        if params_exist:
            chosen_client = client
            chosen_metric = metric
            look_window = lookback
        else:
            chosen_client = form.client_list.data
            chosen_metric = form.metrics_list.data
            look_window = int(form.lookback.data) + 7

        sn = chosen_client
        db_host = info_df[info_df['short_name'] == chosen_client]
        db_host.reset_index(drop=True, inplace=True)
        print(db_host)
        db_host_name = db_host['stats_db'][0]
        passw = db_host['pw'][0]

        stats = create_engine('mysql://datascience:{pw}@{host}/stats_{short}'.
                          format(pw=passw,
                                 host=db_host_name,
                                 short=sn))
        aq = ACCOUNT_QUERY.format(full_date=(datetime.now() -
                                             timedelta(look_window)).strftime('%Y-%m-%d'))

        sq = SITE_DATA_QUERY.format(full_date=(datetime.now() -
                                               timedelta(look_window)).strftime('%Y-%m-%d'))

        control_df = prep_control_chart_data(aq, sq, stats)
        if chosen_metric in ['pct_site_revenue', 'pct_site_orders']:
            toggle = 'site'
        else:
            toggle = 'channel'

        figjs, figdiv = create_trended_plots(control_df, chosen_metric, toggle, sn)
        form = ChannelMetrics()

        config = get_all_site_data()
        conf_info = []
        for blob in config:
            conf_info.append((blob['short_name'], blob['stats_db'], blob['pw']))
        info_df = pd.DataFrame(conf_info)
        info_df.columns = ['short_name', 'stats_db', 'pw']
        info_df.sort_values('short_name', inplace=True)
        client_list = info_df['short_name'].tolist()
        form.client_list.choices = [(i, i) for i in client_list]
        form.metrics_list.choices = [(i, i) for i in ['revenue', 'cost', 'clicks',
                                                      'orders', 'impressions', 'aov',
                                                      'cr', 'ctr', 'cos_m', 'cpc',
                                                      'pct_site_revenue',
                                                      'pct_site_orders']]
        if params_exist:
            form.lookback.default = look_window
            form.client_list.default = client
            form.metrics_list.default = metric

        return render_template('display_moving_control.html',
                               figJS=figjs,
                               figDiv=figdiv,
                               form=form)

    return render_template('base.html', form=form)


@app.route('/client_checkbox', methods=['GET', 'POST'])
def client_checkbox():
    global info_df
    form = ClientEntry()
    config = get_all_site_data()
    conf_info = []
    for blob in config:
        conf_info.append((blob['short_name'], blob['stats_db'], blob['pw']))
    info_df = pd.DataFrame(conf_info)
    info_df.columns = ['short_name', 'stats_db', 'pw']
    info_df.sort_values('short_name', inplace=True)
    client_list = info_df['short_name'].tolist()
    form.client_chooser.choices = [(i, i) for i in client_list]
    if form.validate_on_submit():
        selected_clients = form.client_chooser.data
        print(selected_clients)
        clm = {}
        for i in selected_clients:
            db_host = info_df[info_df['short_name'] == i]
            db_host.reset_index(drop=True, inplace=True)
            db_host_name = db_host['stats_db'][0]
            passw = db_host['pw'][0]
            sn = i
            stats = create_engine('mysql://datascience:{pw}@{host}/stats_{short}'.
                               format(pw=passw,
                                      host=db_host_name,
                                      short=sn))
            clm[i] = run_analysis(stats, sn)

        pd.set_option('display.max_colwidth', -1)
        render_table = pd.DataFrame.from_dict(clm).T
        return render_template('results.html', tables=[render_table.to_html(escape=False)],
                               message='Results dependent on stats availability')

    return render_template('base.html', form=form)


def tier(_row, i):
    if (_row['percentile'] >= 5 * i - 4) & (_row['percentile'] <= 5 * i):
        return 'tier_{i}'.format(i=i)
    else:
        pass


@app.route('/prediction_performance', methods=['GET', 'POST'])
def prediction_performance():
    form = PredictionPerformance()
    sc = SparkContext(appName='R2D2_PerfQuery')
    hc = HiveContext(sc)
    results = hc.sql(PERF_PRED_QUERY.format(date=(datetime.now() - timedelta(14)).strftime(
        '%Y-%m-%d')))
    prediction_stats = results.toPandas()

    client_list = prediction_stats['site_id'].tolist()
    date_list = prediction_stats['prediction_date'].tolist()

    form.client_list.choices = [(i, i) for i in client_list]
    form.date_list.choices = [(i, i) for i in date_list]
    if form.validate_on_submit():
        tiers = []
        for i in range(1, 21):
            tiers1 = prediction_stats.apply(lambda row: tier(row, i), axis=1)
            tiers.append(tiers1)

        flatten = lambda tiers: [item for sublist in tiers for item in sublist]
        tiers = flatten(tiers)
        tiers = [x for x in tiers if x != None]
        prediction_stats.sort_values(['percentile', 'prediction_date'],
                                     ascending=True, inplace=True)
        prediction_stats['tier'] = tiers

        tiers_summary = prediction_stats.groupby(['site_id', 'prediction_date',
                                                  'model_id', 'tier']).\
            agg({'cost': 'sum', 'revenue': 'sum', 'cum_percent_cost': 'max',
                 'cum_percent_revenue': 'max'}).reset_index(drop=False)

        return render_template('results.html', tables=[tiers_summary.to_html(escape=False)],
                               message='Results dependent on stats availability')

    return render_template('base.html', form=form)


@app.route('/', methods=['GET', 'POST'])
def root_document():
    form = MainForm()
    if form.rank_models.data:
        return redirect(url_for('ranking_rmse'))
    if form.compare_models.data:
        return redirect(url_for('comparison'))
    if form.see_importance.data:
        return redirect(url_for('importance'))
    if form.daily_changes.data:
        return redirect(url_for('daily_snapshot_clients'))
    if form.adjust_threshold.data:
        return redirect(url_for('threshold_sliders'))
    if form.run_simulation.data:
        return redirect(url_for('simulation'))
    if form.channel_performance.data:
        return redirect(url_for('channel_metrics'))
    if form.client_checkbox.data:
        return redirect(url_for('client_checkbox'))
    if form.prediction_performance.data:
        return redirect(url_for('prediction_performance'))

    return render_template('base.html', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
