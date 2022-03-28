def init_equity():
	query = """
		SELECT "ISINCode", "Country", "EquitySize", "EquitySector", "ESG", "EquityFactorDes", "EquityStrategyDes",
		"DistributionIndicator", "ETPIssuerName", "FundCurrency", "Cost", "IndexProvider"
		FROM gcp_class WHERE "AssetClass" = 'Equity' 
	 """
	return query 

def init_fi():
	query = """
		SELECT "ISINCode", "Country", "FixedIncomeType", "FixedIncomeRatingGroup", "ESG", "FixedIncomeMaturityGroup", 
		"FixedIncomeDominateCurrencyGroup", "FixedIncomeFactorDes", "FixedIncomeStrategyDes",
		"DistributionIndicator", "ETPIssuerName", "FundCurrency", "Cost", "IndexProvider"
		FROM gcp_class WHERE "AssetClass" = 'Fixed Income' 
		"""
	return query

def init_commodity():
	query = """
		SELECT "ISINCode", "CommodityUnderlying", "CommoditySpotForward", "CommodityStrategyDes",
		"DistributionIndicator", "ETPIssuerName", "FundCurrency", "Cost", "IndexProvider"
		FROM gcp_class WHERE "AssetClass" = 'Commodity' 
		"""
	return query 


def init_currency():
	query = """
		SELECT "ISINCode", "CurrencyBucket1",
		"DistributionIndicator", "ETPIssuerName", "FundCurrency", "Cost", "IndexProvider"
		FROM gcp_class WHERE "AssetClass" = 'Currency' 
		"""
	return query

def init_structured():
	query = """
		SELECT "ISINCode", "StructuredMultiple", "StructuredTracking",
		"DistributionIndicator", "ETPIssuerName", "FundCurrency", "Cost", "IndexProvider"
		FROM gcp_class WHERE "AssetClass" = 'Structured' 
		"""
	return query

def init_alt():
	query = """
		SELECT "ISINCode", "SubAssetClass",
		"DistributionIndicator", "ETPIssuerName", "FundCurrency", "Cost", "IndexProvider"
		FROM gcp_class WHERE "AssetClass" = 'Alternative & Multi-Assets' 
		"""
	return query

def init_thematic():
	query = """
		SELECT "ISINCode", "EquitySubSector",
		"DistributionIndicator", "ETPIssuerName", "FundCurrency", "Cost", "IndexProvider"
		FROM gcp_class WHERE "AssetClass" = 'Equity' AND "EquityIsThematic" =True
		"""
	return query 

def initialise(asset_class, exchange):
	if asset_class == 'Equity':
		query = init_equity()
	elif asset_class == 'Fixed Income':
		query = init_fi()
	elif asset_class == 'Commodity':
		query = init_commodity()
	elif asset_class == 'Currency':
		query = api.init_currency()
	elif asset_class == 'Structured':
		query = api.init_structured()
	elif asset_class == 'Alternative & Multi-Assets':
		query = api.init_alt()
	elif asset_class == 'Thematic':
		query = init_thematic()
	else:
		return

	query += """ AND "Exchange" = '"""+ exchange +"""' """
	return query


def get_exchanges():
	query = """ SELECT "Flag", "Country", "Exchange" FROM (
			(SELECT DISTINCT("Exchange"), COUNT("Ticker") FROM tickers GROUP BY 1) AS exchanges
			LEFT JOIN country_list
			USING("Exchange")
			) exchange_map ORDER BY count DESC"""
	return query

def get_etfs_list(exchange=None, include_class = False):
	if include_class == False:
		query = """ SELECT "ISINCode", "ExchangeTicker", "FundName" FROM gcp_funds """
	else: 
		query = """ SELECT "ISINCode", "ExchangeTicker", "FundName", "Sector" FROM gcp_funds """
	if exchange is not None:
		query += """ WHERE "Exchange" = '""" + exchange + """' """

	query += """ ORDER BY "AUM_USD" DESC NULLS LAST"""
	return query


def get_by_isin(isins, table_name, col_names = None, limit = None):
	isin_lst = '\', \''.join(isins)
	if col_names is not None:
		cols = '\", \"'.join(col_names)
		query = """ SELECT \"""" + cols + """\" FROM """ +table_name+ """ WHERE "ISINCode" IN ('""" + isin_lst + """')"""
	else:
		query = """ SELECT * FROM """ +table_name+ """ WHERE "ISINCode" IN ('""" + isin_lst + """')"""

	if limit is not None:
		query = query + """ LIMIT """ + str(limit)

	return query

def get_by_ticker(tickers, table_name, col_names = None):
	ticker_lst = '\', \''.join(tickers)
	if col_names is not None:
		cols = '\", \"'.join(col_names)
		query = """ SELECT \"""" + cols + """\" FROM """ +table_name+ """ WHERE "ExchangeTicker" IN ('""" + ticker_lst + """')"""
	else:
		query = """ SELECT * FROM """ +table_name+ """ WHERE "Ticker" IN ('""" + ticker_lst + """')"""

	return query

def search(txt_lst):
	search_arr = ['\'%' + i + '%\'' for i in txt_lst]

	query = """ SELECT "ISINCode" FROM 
		(SELECT "ISINCode", LOWER("FundName") AS "Name" FROM gcp_funds) AS names
		WHERE "Name" LIKE any(array[ """ + ', '.join(search_arr) + """ ])"""

	return query


def get_fundflows(isins):
	query = get_by_isin(isins, 'calc_fundflow_period', ["ISINCode", "Description", "Currency", "flow_local", "flow_USD", "flow_EUR", "flow_GBP"])
	query = """SELECT * FROM ( ("""+ query + """ ) AS flows LEFT JOIN (SELECT "ISINCode", "AUM", "AUM_USD", "AUM_EUR", "AUM_GBP" FROM latest_nav_shares) AS aums USING("ISINCode") ) AS flow_map"""
	return query


def get_div(isins):
	query_funds = get_by_isin(isins, 'funds', ["ISINCode", "DistributionIndicator", "CashFlowFrequency"])
	query_div = get_by_isin(isins, 'latest_div')

	query = """ SELECT "ISINCode","DistributionIndicator", "CashFlowFrequency",
				"exDivDate", "Currency", "Dividend", "Yield", "DivGrowth", "DivGrowthPct"
				 FROM ( ( """ + query_funds + """) AS funds
			LEFT JOIN latest_div
			USING("ISINCode")
			) AS div_map
			"""
	return query



def get_similar_etfs(isin):
	query = get_by_isin(isin, 'calc_similaretfs')

	query = """ WITH etfs AS ( """ + query + """)
		SELECT "des1" AS "Description" ,unnest("isin1") AS "ISINCode" FROM(
			SELECT "des1" ,"isin1" FROM etfs
			UNION
			SELECT "des2", "isin2" FROM etfs
			UNION
			SELECT "des3", "isin3" FROM etfs
			UNION
			SELECT "des4", "isin4" FROM etfs
			UNION
			SELECT "des5", "isin5" FROM etfs
			UNION
			SELECT "des6", "isin6" FROM etfs
			UNION
			SELECT "des7", "isin7" FROM etfs
			UNION
			SELECT "des8", "isin8" FROM etfs
		) AS nested_etfs """ 

	return query

def get_top5_holding(isin, include_isin = False):
	cols = []
	if include_isin == True:
		cols = ['ISINCode'] + cols
	query = get_by_isin(isin, 'calc_holding_all', ["ISINCode", "TimeStamp" ,"InstrumentDescription", "Weight", "Country"])

	query = """	SELECT "ISINCode", "InstrumentDescription", "Weight", "Country" FROM (
				SELECT *, Last_VALUE("TimeStamp") OVER (PARTITION BY "ISINCode" RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS latest
				FROM (""" + query + """ AND "Rank" <= 10 ) AS holding  
				) AS holding_rank WHERE "TimeStamp" = latest """

	query = """ SELECT * FROM (
				( """ + query + """) AS holding_top10
				LEFT JOIN (SELECT "Country", "Flag" FROM country_list ) AS flags
				USING("Country")
			) AS flag_map """
	return query

def get_navs_weekly(isins):
	query = get_by_isin(isins, 'nav_shares_map', ["ISINCode","TimeStamp", "NAV"])
	query += """ AND "IsWeek" = TRUE ORDER BY "ISINCode" ,"TimeStamp" ASC """
	return query 

def get_1y_perf(isins):
	query = get_by_isin(isins, 'calc_return_period', ["ISINCode", "Return"])
	query = """ SELECT * FROM (
				(""" + query + """ AND "Description" = '1Y' AND "Type"='Cumulative') AS perf
				LEFT JOIN (SELECT "ISINCode", "FundName" FROM funds) AS funds
				USING("ISINCode")
			) AS perf_map """
	return query

def get_details(isin):

	q_funds = get_by_isin(isin, 'funds', ["ISINCode" ,"LaunchDate", "CashFlowFrequency","FundCompany","ETPIssuerName", "ShortIssuerName", "Custodian", "LegalStructure", "Domicile", "UCITS"])
	
	q_shares = get_by_isin(isin, 'latest_nav_shares', [ "ISINCode","Shares"])

	q_index = get_by_isin(isin, 'class_copy', ["ISINCode","IndexAgg","IndexProvider", "RebalanceFrequency", "Backing", "Synthetic", "CurrencyHedge", "HedgeFrom", "HedgeTo"])

	q_index = """SELECT * FROM (
			(SELECT "ISINCode","IndexAgg","IndexProvider", "RebalanceFrequency", ("Backing" || CASE "Backing" WHEN 'Synthetic' THEN ' (' || "Synthetic" || ')' ELSE '' END) AS "Replication",
			(CASE "CurrencyHedge" WHEN True THEN 'Yes (from ' || "HedgeFrom" || ' to ' || "HedgeTo" || ')' ELSE 'No' END) AS "CurrencyHedge"
			FROM (""" + q_index + """) AS class_isin) AS classification
			LEFT JOIN (SELECT "IndexAgg", "IndexDescription" FROM indices) AS indices
			USING("IndexAgg")
		) AS index_map """

	q_div = get_by_isin(isin, 'latest_div', ["ISINCode", "exDivDate", "Currency", "Dividend", "Yield"])
	q_tracking = get_by_isin(isin, """"trackingErrors" """, ["ISINCode", "TrackingError3Y" ])
	
	q_vol = get_by_isin(isin, 'calc_vol_period', ["ISINCode", "Volatility"])
	q_vol += """ AND "Description" = '3Y' """

	q_dd = get_by_isin(isin, 'calc_maxdrawdown_period', ["ISINCode", "Drawdown", "DateDrawdown"])
	q_dd += """ AND "Description" = '3Y' """


	query = """ SELECT * FROM (
			(""" + q_funds + """ ) AS q_funds 
			LEFT JOIN ( """ +q_shares+""") AS q_shares
			USING ("ISINCode")
			LEFT JOIN ( """ + q_index + """ ) as q_index
			USING("ISINCode")
			LEFT JOIN ( """ + q_div + """) as q_div
			USING("ISINCode")
			LEFT JOIN ( """ +q_tracking + """) as q_tracking
			USING("ISINCode")
			LEFT JOIN (""" + q_vol + """ ) as q_vol
			USING("ISINCode")
			LEFT JOIN (""" +q_dd + """) as q_dd
			USING("ISINCode")
			) AS all_map """
	return query


def get_listings(isin):
	q_stats = get_by_isin(isin, 'latest_ticker', ["Ticker", "Price", "Volume3M"])
	q_ticker = get_by_isin(isin, 'tickers', ["Ticker", "TradingCurrency","ExchangeTicker", "Exchange", "SEDOL", "CUSIP", "Valoren", "ExchangeHour"])


	query = """ SELECT ("Flag" || ' ' || "Exchange") AS "Exchange", "ExchangeTicker", "TradingCurrency", "Price", "Volume3M", "SEDOL", "CUSIP", "Valoren", "ExchangeHour" FROM (
			(""" + q_stats + """) AS stats
			LEFT JOIN
			(""" + q_ticker + """) as ticker
			USING("Ticker")
			LEFT JOIN (SELECT "Exchange", "Flag" FROM country_list) as flags
			USING("Exchange")
			) AS all_map ORDER BY "Volume3M" DESC """
	return query


def get_tr(isins, start_date = None, end_date=None):
	query = get_by_isin(isins, 'total_return', ["ISINCode", "TimeStamp", "TotalReturn"])

	if start_date is not None:
		query += """ AND "TimeStamp" >= '""" +  str(start_date) + """' """

	if end_date is not None:
		query += """ AND "TimeStamp" <= '""" +  str(end_date) + """' """

	return query

def get_nav(isins, start_date=None, end_date=None):
	query = get_by_isin(isins, 'nav_shares_map', ["ISINCode", "TimeStamp", "NAV"])

	if start_date is not None:
		query += """ AND "TimeStamp" >= '""" +  str(start_date) + """' """
	
	if end_date is not None:
		query += """ AND "TimeStamp" <= '""" +  str(end_date) + """' """

	query += """ ORDER BY "ISINCode", "TimeStamp" ASC """
	return query


def get_vol_period(isins):
	query = get_by_isin(isins, 'calc_vol_period', ["ISINCode", "Description", "Volatility"])
	query += """ AND "Type" = 'Relative Period' """

	return query

def get_drawdown_period(isins):
	query = get_by_isin(isins, 'calc_maxdrawdown_period', ["ISINCode", "Description" ,"Drawdown", "DateDrawdown"])
	query += """ AND "Type" = 'Period Drawdown' """

	return query

def get_holding_all(isin):

	query = get_by_isin(isin, 'calc_holding_all', ["ISINCode","TimeStamp","InstrumentDescription", "Country", "Sector", "Weight"])
	query = """ 
			SELECT * FROM (
				SELECT *, LAST_VALUE("TimeStamp") OVER(PARTITION BY "ISINCode" RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) "Latest" FROM (""" \
				+query + """) AS holding 
			) holding_latest WHERE "TimeStamp" = "Latest" """

	query = """ SELECT "ISINCode","InstrumentDescription", "Country" ,"Flag", "Sector", "Weight" FROM (
					( """ + query + """) AS holding 
					LEFT JOIN country_list
					USING("Country")
					) AS flag_map """

	return query

def get_holding_type(isin):

	query = get_by_isin(isin, 'calc_holding_type', ["ISINCode", "TimeStamp","HoldingType", "HoldingName", "Weight"])
	query = """ 
			SELECT * FROM (
				SELECT *, LAST_VALUE("TimeStamp") OVER(PARTITION BY "ISINCode" RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) "Latest" FROM ("""\
				+query + """ AND "Rank"<=10) AS holding 
			) holding_latest WHERE "TimeStamp" = "Latest" """


	query = """ SELECT "ISINCode","HoldingType", "HoldingName", "Weight", "Flag" FROM (
				( """ + query + """ ) AS holding
				LEFT JOIN country_list
				ON holding."HoldingName" = country_list."Country"
				) AS flag_map """
 
	return query

def get_similar_etfs_details(isins):

	query_funds = get_by_isin(isins, 'funds', ["ISINCode", "FundName","DistributionIndicator", "TotalExpenseRatio"])

	query_aum = get_by_isin(isins, 'latest_nav_shares', ["ISINCode", "Currency", "AUM", "AUM_USD"])

	query_perf = get_by_isin(isins, 'calc_return_period', ["ISINCode", "Return"])
	query_perf += """ AND "Description" = 'YTD' """ 

	query = """SELECT "ISINCode", "FundName","DistributionIndicator", "TotalExpenseRatio", "Currency",  "AUM", "Return" FROM (
				(""" + query_funds + """ ) AS funds
				LEFT JOIN (""" + query_aum + """) AS aum
				USING("ISINCode")
				LEFT JOIN (""" + query_perf + """ ) AS perf
				USING("ISINCode")
			) AS perf_map ORDER BY "TotalExpenseRatio" ASC, "AUM_USD" DESC  """ 
	return query


def get_equity_indices_names():
	return """SELECT "Ticker", "Flag", "Name" FROM (
			(SELECT *, ROW_NUMBER() OVER() AS row_order FROM "equity_indices_list" ) as indices
			LEFT JOIN (SELECT "Country", "Flag" FROM "country_list") AS countries
			USING("Country")
		) flag_map ORDER BY row_order"""

def get_equity_indices(tickers, start_date = None, end_date=None):
	ticker_lst = '\', \''.join(tickers)
	query = """ SELECT * FROM "equity_indices" WHERE "Ticker" IN ('""" + ticker_lst + """')"""
	if start_date is not None:
		query += """ AND "Timestamp" >= ' """ + str(start_date) + """' """

	if end_date is not None:
		query += """ AND "Timestamp" <= ' """ + str(end_date) + """' """

	return query


def get_fundflow(isins, currency_col ,start_date = None, end_date=None):
	query = get_by_isin(isins, 'calc_fundflow_monthly', ['ISINCode', 'TimeStamp', currency_col])

	if start_date is not None:
		query += """ AND "TimeStamp" >= '""" +  str(start_date) + """' """
	
	if end_date is not None:
		query += """ AND "TimeStamp" <= '""" +  str(end_date) + """' """

	query += """ ORDER BY "ISINCode", "TimeStamp" ASC """

	return query

def get_benchmark_port():
	return """SELECT DISTINCT("Portfolio") FROM portfolios ORDER BY "Portfolio" """

def get_benchmark_weight(name):
	return """ SELECT "ISINCode", "Weight" FROM portfolios WHERE "Portfolio" = '""" + eval('name') + """' """

def get_sector(isins):
	query = get_by_isin(isins, 'gcp_funds', ['ISINCode', 'FundName', 'AssetClass', 'Sector'])
	query = """select DISTINCT("ISINCode"), "FundName", ("AssetClass" || ' - ' || "Sector") AS "Sector" 
	from ( """ +query + """) as sectors """
	return query


def get_usd_prices(isins):
	nav = get_by_isin(isins, 'nav_shares_map', ["ISINCode", "TimeStamp", "Currency", "NAV"])
	tr = get_by_isin(isins, 'total_return', ["ISINCode", "TimeStamp", "TotalReturn"])

	query = """SELECT "ISINCode", "TimeStamp", "NAV_USD", "TR_USD", "FX_GBP", "FX_EUR" FROM (
		(SELECT "ISINCode","TimeStamp", "Currency", "NAV", "TotalReturn", "FX",
		CASE 
			WHEN "Currency" = 'USD' THEN "NAV"
			WHEN "Currency" = 'GBX' THEN "NAV" * "FX" / 100
			WHEN "Currency" = 'GBP' OR "Currency" = 'EUR' OR "Currency" = 'AUD' OR "Currency" = 'NZD' THEN "NAV" * "FX"
			ELSE "NAV"/"FX" END "NAV_USD",  
		CASE 
			WHEN "Currency" = 'USD' THEN "TotalReturn"
			WHEN "Currency" = 'GBX' THEN "TotalReturn" * "FX" / 100
			WHEN "Currency" = 'GBP' OR "Currency" = 'EUR' OR "Currency" = 'AUD' OR "Currency" = 'NZD' THEN "TotalReturn" * "FX"
			ELSE "TotalReturn"/"FX" END "TR_USD"   
		FROM (
			SELECT * FROM (
				(SELECT *,EXTRACT(isodow FROM "TimeStamp") AS "WeekDay",
					CASE
						WHEN "Currency" = 'GBX' THEN 'GBP='
						ELSE "Currency"||'=' END "FXTicker" FROM ( """ + nav + """ ) as nav_query ) AS a
				LEFT JOIN (""" + tr + """) as b
				USING("ISINCode", "TimeStamp")
				LEFT JOIN (SELECT "TimeStamp", "FXTicker", "FX" FROM fxes) AS c
				USING ("TimeStamp", "FXTicker")
				) AS map where "WeekDay" <=5 
			) AS period_map 
		) extract_map 
		LEFT JOIN (SELECT "TimeStamp", "FX" AS "FX_GBP" FROM fxes WHERE "FXTicker" = 'GBP=') AS gbp
		USING("TimeStamp")
		LEFT JOIN (SELECT "TimeStamp", "FX" AS "FX_EUR" FROM fxes WHERE "FXTicker" = 'EUR=') AS eur
		USING("TimeStamp")
	) AS fx_map ORDER BY "ISINCode", "TimeStamp"  """

	return query


def get_usd_div(isins):
	div = get_by_isin(isins, 'calc_div_yield', ["ISINCode", "TimeStamp", "Currency","Dividend"])

	query = """SELECT "TimeStamp", "ISINCode", "DIV_USD", "FX_GBP", "FX_EUR" FROM (
			(SELECT "ISINCode","TimeStamp", "Currency", "Dividend", "FX",
		        CASE 
		            WHEN "Currency" = 'USD' THEN "Dividend"
		            WHEN "Currency" = 'GBX' THEN "Dividend" * "FX" / 100
		            WHEN "Currency" = 'GBP' OR "Currency" = 'EUR' OR "Currency" = 'AUD' OR "Currency" = 'NZD' THEN "Dividend" * "FX"
		            ELSE "Dividend"/"FX" END "DIV_USD"   
		    FROM (
		        SELECT * FROM (
					(SELECT *,
						EXTRACT(isodow FROM "TimeStamp") AS "WeekDay",
						CASE
							WHEN "Currency" = 'GBX' THEN 'GBP='
							ELSE "Currency"||'=' END "FXTicker" FROM ( """ + div + """) as div ) AS a
					LEFT JOIN (SELECT "TimeStamp", "FXTicker", "FX" FROM fxes) AS c
					USING ("TimeStamp", "FXTicker")
		            ) AS map where "WeekDay" <=5 
		    	) AS period_map 
			) extract_map 
			LEFT JOIN (SELECT "TimeStamp", "FX" AS "FX_GBP" FROM fxes WHERE "FXTicker" = 'GBP=') AS gbp
			USING("TimeStamp")
			LEFT JOIN (SELECT "TimeStamp", "FX" AS "FX_EUR" FROM fxes WHERE "FXTicker" = 'EUR=') AS eur
			USING("TimeStamp")
		) AS fx_map ORDER BY "ISINCode", "TimeStamp" """

	return query

