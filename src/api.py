def init_equity():
	query = """
		SELECT "ISINCode", "Country", "EquitySize", "EquitySector", "IsESG", "EquityFactorDes", "EquityStrategyDes" 
		FROM gcp_class WHERE "AssetClass" = 'Equity' 
	 """
	return query 

def init_fi():
	query = """
		SELECT "ISINCode", "Country", "FixedIncomeType", "FixedIncomeRatingGroup", "IsESG", "FixedIncomeMaturityGroup", 
		"FixedIncomeDominateCurrencyGroup", "FixedIncomeFactorDes", "FixedIncomeStrategyDes" 
		FROM gcp_class WHERE "AssetClass" = 'Fixed Income' 
		"""
	return query

def init_commodity():
	query = """
		SELECT "ISINCode", "CommodityUnderlying", "CommoditySpotForward", "CommodityStrategyDes"
		FROM gcp_class WHERE "AssetClass" = 'Commodity' 
		"""
	return query 


def init_currency():
	query = """
		SELECT "ISINCode", "CurrencyBucket1"
		FROM gcp_class WHERE "AssetClass" = 'Currency' 
		"""
	return query

def init_structured():
	query = """
		SELECT "ISINCode", "StructuredMultiple", "StructuredTracking"
		FROM gcp_class WHERE "AssetClass" = 'Structured' 
		"""
	return query

def init_alt():
	query = """
		SELECT "ISINCode", "SubAssetClass"
		FROM gcp_class WHERE "AssetClass" = 'Alternative & Multi-Assets' 
		"""
	return query

def init_thematic():
	query = """
		SELECT "ISINCode", "EquitySubSector"
		FROM gcp_class WHERE "AssetClass" = 'Equity' AND "EquityIsThematic" =True
		"""
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


def get_fundflows(isins):
	query = get_by_isin(isins, 'calc_fundflow_period', ["ISINCode", "Description", "Currency", "flow_local", "flow_USD", "flow_EUR", "flow_GBP"])
	query = """SELECT * FROM ( ("""+ query + """ ) AS flows LEFT JOIN (SELECT "ISINCode", "AUM", "AUM_USD", "AUM_EUR", "AUM_GBP" FROM latest_nav_shares) AS aums USING("ISINCode") ) AS flow_map"""
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

def get_top5_holding(isin):
	query = get_by_isin(isin, 'calc_holding_all', ["InstrumentDescription", "Weight", "Country"])

	query = """ SELECT * FROM (
				( """ + query + """ LIMIT 10) AS holding
				LEFT JOIN (SELECT "Country", "Flag" FROM country_list ) AS flags
				USING("Country")
			) AS flag_map """
	return query

def get_navs_weekly(isins):
	query = get_by_isin(isins, 'nav_shares_map', ["ISINCode","TimeStamp", "NAV"])
	query += """ AND "IsWeek" = TRUE ORDER BY "ISINCode" ,"TimeStamp" ASC """
	return query 

def get_1m_perf(isins):
	query = get_by_isin(isins, 'calc_return_period', ["ISINCode", "Return"])
	query = """ SELECT * FROM (
				(""" + query + """ AND "Description" = '1M') AS perf
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


def get_tr(isins, start_date):
	query = get_by_isin(isins, 'total_return', ["ISINCode", "TimeStamp", "TotalReturn"])
	query += """ AND "ISINCode" >= '""" +  start_date + """' """
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
	query_max = get_by_isin(isin, 'calc_holding_all', ["TimeStamp"])
	query_max = """SELECT MAX("TimeStamp") FROM ( """ + query_max + """ ) AS max_date """

	query = get_by_isin(isin, 'calc_holding_all', ["InstrumentDescription", "Country", "Sector", "Weight"])
	query += """ AND "TimeStamp" = (""" + query_max + """)"""

	query = """ SELECT "InstrumentDescription", "Country" ,"Flag", "Sector", "Weight" FROM (
					( """ + query + """) AS holding 
					LEFT JOIN country_list
					USING("Country")
					) AS flag_map """

	return query

def get_holding_type(isin):
	query_max = get_by_isin(isin, 'calc_holding_type', ["TimeStamp"])
	query_max = """SELECT MAX("TimeStamp") FROM ( """ + query_max + """ ) AS max_date """

	query = get_by_isin(isin, 'calc_holding_type', [ "HoldingType", "HoldingName", "Weight"])
	query += """ AND "Rank"<=10 AND "TimeStamp" = (""" + query_max + """)"""

	query = """ SELECT "HoldingType", "HoldingName", "Weight", "Flag" FROM (
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









