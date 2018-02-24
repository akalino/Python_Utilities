import ast
import io
import json
import tempfile

from datetime import datetime, timedelta
from sqlalchemy import create_engine

import boto
import findspark
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

findspark.init()

from flask import Flask, render_template, redirect, url_for
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import Form
from pyspark import SparkContext, HiveContext
from site_data import get_all_site_data
from wtforms import SubmitField, SelectField, StringField
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


def lookback_validator(form, field):
    if int(field.data) < 14:
        raise ValidationError('The lookback must be longer than 14 days')


class MainForm(Form):
    rank_models = SubmitField('Rank All Models')
    compare_models = SubmitField('Compare Models')
    see_importance = SubmitField('Variable Importance')
    daily_changes = SubmitField('Daily Highlights')
    adjust_threshold = SubmitField('Adjust Thresholds')
    run_simulation = SubmitField('Simulation Bake Off')
    channel_performance = SubmitField('Channel Performance')


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

    jd.drop(['sum_orders', 'sum_clicks', 'sum_impressions', 'sum_revenue', 'sum_cost'], axis=1, inplace=True)

    jd = jd.dropna()
    jd.reset_index(drop=True, inplace=True)

    return jd


def create_trended_plots(_df, _variable, _level):
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
        from pylab import rcParams
        rcParams['figure.figsize'] = 15, 7
        site_daf.plot(x='Date',
            ylim=(0, max(site_daf[_metric] * 1.1)),
            color='black',
            marker='o',
            title='Google Shopping as a Pct of Site {var}'.format(var=_variable.title()))

        var_mean = site_daf[_metric].mean()
        var_sd = site_daf[_metric].std()

        plt.axhline(y=var_mean, linewidth=1, color='blue')
        plt.axhline(y=var_mean + var_sd*1, linewidth=1.5, color='green')
        plt.axhline(y=var_mean - var_sd*1, linewidth=1.5, color='green')
        plt.axhline(y=var_mean + var_sd*2, linewidth=2, color='yellow')
        plt.axhline(y=var_mean - var_sd*2, linewidth=2, color='yellow')
        plt.axhline(y=var_mean + var_sd*3, linewidth=2.5, color='red')
        plt.axhline(y=var_mean - var_sd*3, linewidth=2.5, color='red')
        plt.ylabel('Daily Percent of Site {var}'.format(var=_variable.title()))
        f = tempfile.NamedTemporaryFile(
            dir='static/temp',
            suffix='.png',
            delete=False)

        plt.savefig(f)
        f.close()
        plotPng = f.name.split('/')[-1]

    elif _level == 'channel':
        from pylab import rcParams
        rcParams['figure.figsize'] = 15, 7
        channel_df.plot(x='Date',
            ylim=(0, max(channel_df[_variable] * 1.1)),
            colors=['black', 'orange'],
            marker='o',
            title='Google Shopping {var} Trended'.format(var=_variable.title()))

        var_mean = channel_df[_variable].mean()
        var_sd = channel_df[_variable].std()

        plt.axhline(y=var_mean, linewidth=1, color='blue')
        plt.axhline(y=var_mean + var_sd*1, linewidth=1.5, color='green')
        plt.axhline(y=var_mean - var_sd*1, linewidth=1.5, color='green')
        plt.axhline(y=var_mean + var_sd*2, linewidth=2, color='yellow')
        plt.axhline(y=var_mean - var_sd*2, linewidth=2, color='yellow')
        plt.axhline(y=var_mean + var_sd*3, linewidth=2.5, color='red')
        plt.axhline(y=var_mean - var_sd*3, linewidth=2.5, color='red')
        plt.ylabel('Google Shopping Daily {var}'.format(var=_variable.title()))
        f = tempfile.NamedTemporaryFile(
            dir='static/temp',
            suffix='.png',
            delete=False)

        plt.savefig(f)
        f.close()
        plotPng = f.name.split('/')[-1]

    return plotPng

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
    form = ProductDisplay()
    form.product_list.choices = [(i, i+1) for i in range(len(site_df))]
    if form.product_list.data:
        if form.product_list.data != 'None':
            print(form.product_list.data)
            print(site_df['image_url'][int(form.product_list.data)])
            return render_template('display_products.html', form=form,
                                   img_link=site_df['image_url'][int(form.product_list.data)],
                                   prod_title=site_df['title'][int(form.product_list.data)],
                                   product_id=site_df['product_id'][int(form.product_list.data)])
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

    if form.validate_on_submit():
        chosen_client = form.client_list.data
        db_host = info_df[info_df['short_name'] == chosen_client]
        db_host.reset_index(drop=True, inplace=True)
        print(db_host)
        db_host_name = db_host['stats_db'][0]
        passw = db_host['pw'][0]
        look_window = int(form.lookback.data) + 7
        sn = chosen_client
        stats = create_engine('mysql://datascience:{pw}@{host}/stats_{short}'.
                          format(pw=passw,
                                 host=db_host_name,
                                 short=sn))
        aq = ACCOUNT_QUERY.format(full_date=(datetime.now() -
                                             timedelta(look_window)).strftime('%Y-%m-%d'))

        sq = SITE_DATA_QUERY.format(full_date=(datetime.now() -
                                               timedelta(look_window)).strftime('%Y-%m-%d'))

        control_df = prep_control_chart_data(aq, sq, stats)
        chosen_metric = form.metrics_list.data
        if chosen_metric in ['pct_site_revenue', 'pct_site_orders']:
            toggle = 'site'
        else:
            toggle = 'channel'

        img_io = create_trended_plots(control_df, chosen_metric, toggle)
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

        return render_template('display_control_chart.html', plotPng=img_io, form=form)

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

    return render_template('base.html', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
