import pandas as pd
import altair as alt
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from src.style import color_palette

@st.cache(allow_output_mutation=True)
def performance_line_simple(data, y_col, num_format):
	nearest = alt.selection(type='single', nearest=True, on='mouseover',
			fields=['Dates'], empty='none')

	base = alt.Chart(data).mark_line(color='#304FFE').encode(
			x=alt.X('Dates:T', title=None ,axis=alt.Axis(format = ("%b-%Y"))),
			y = alt.Y(y_col + ':Q', title=None, axis=alt.Axis(format= '.1f'), scale=alt.Scale(zero=False)),
		)


	selectors = alt.Chart(data).mark_point().encode(
			x='Dates:T',
			opacity=alt.value(0),
		).add_selection(
			nearest
		)

	points = base.mark_point().encode(
			opacity=alt.condition(nearest, alt.value(1), alt.value(0))
		)

	text = base.mark_text(align='left', dx=-50, dy=10).encode(
			text=alt.condition(nearest, alt.Text(y_col +':Q', format=num_format), alt.value(' ')),
		)

	rules = alt.Chart(data).mark_rule(color='#E7E8F0').encode(
			x = 'Dates:T'
		).transform_filter(
		nearest)

	return alt.layer(
			base, selectors, points, text, rules
			).properties(
			height = 400
			)


@st.cache
def draw_radar_graph(data1, data2, categories, legend1, legend2):
	fig = go.Figure()
	rng = ['#123FED', '#FFC400']

	fig.add_trace(go.Scatterpolar(
		      r=data1,
		      theta=categories,
		      fill='toself',
		      name=legend1,
		      marker_color = rng[0],
		      hovertemplate = "%{theta}: %{r}"
			)
			)
	fig.add_trace(go.Scatterpolar(
      		r=data2,
      		theta=categories,
      		fill='toself',
      		name=legend2,
      		marker_color = rng[1],
      		hovertemplate = "%{theta}: %{r}"
		))
	fig.update_layout(
			margin = dict(l=50, r =50, t=0, b=80),
		  	polar=dict(
		    	radialaxis=dict(
		    		visible=True,
			      	range=[0, 100],
		    	)),
		  	legend=dict(
		  		orientation="h",
		  		y = 1,
		  		x = 1,
		  		yanchor='bottom',
		  		xanchor='center'
		  		),
		  	font_family = 'Sans-serif',
		  	font_size = 14,
		  	title_font_family='Sans-serif'
  			
		  #showlegend=True, 	  
		)
	fig.layout.plot_bgcolor = '#F7F8FA'
	return fig


#@st.cache(allow_output_mutation=True)
def draw_grouped_bar_vertical(data, x_name, y_name, row_name, dom , sort_order):
	tooltips = [y_name, row_name, alt.Tooltip(x_name, format = '.2~s')]

	bars = alt.Chart(data).mark_bar().encode(
				x = alt.X(x_name +':Q', axis = alt.Axis(format='~s')),
				y = alt.Y(y_name,title=None),
				color = alt.Color(y_name, scale=alt.Scale(domain=dom)),
				row = alt.Row(row_name+':N', sort=sort_order, title=None, spacing=5),
				tooltip = tooltips
			).configure_header(
				labelOrient = "right"
			).configure_axisY(
			disable=True
			)

	return bars

@st.cache(allow_output_mutation=True)
def draw_top_holding_graph(data):
	base = alt.Chart(data).transform_calculate(
			combined=alt.datum.Flag + '    ' + alt.datum.Name
			).encode(
				x = alt.X('Weight:Q', axis = alt.Axis(format='.1%'), scale=alt.Scale(type='linear')),
				y = alt.Y('combined:N', sort='-x')			
			).properties(
				width = 400,
				height = alt.Step(40)
				)
	bars = base.mark_bar(color='#123FED').encode(
		tooltip = ['Name', 'Country', alt.Tooltip('Weight', format = '.2%')]
		).properties(height=alt.Step(30))

	text = base.mark_text(align='right', color='white').encode(
		text = alt.Text('Weight:N', format='.1%')
		)

	return bars +  text


def draw_holding_type(data, y_col, title, y_label_width=120):
	
	axis = alt.Axis(labelAlign = 'left', labelColor='black', labelPadding=y_label_width, labelLimit=y_label_width)

	base = alt.Chart(data).mark_bar(color='#123FED').encode(
		x = alt.X('Weight:Q', axis=alt.Axis(format='.1%')),
      	y = alt.Y(y_col+':N', sort='-x', axis=axis),
      	#y = alt.Y(y_col + ':N', sort='-x', axis=alt.Axis(labelAlign='left')),
      	tooltip = ['Holding', alt.Tooltip('Weight', format='.1%')]
		).properties(height = alt.Step(35))


	text2 = base.mark_text(align='right',dx=-5, color='White').encode(
		text = alt.Text('Weight:Q', format='.1%'))

	return (base + text2).properties(title = title)



@st.cache(allow_output_mutation=True)
def performance_graph(data, y_col, dom, names_seq, num_format, color_col = 'Type'):
	main_palette, _ = color_palette()
	names_seq['Color'] = main_palette[:len(names_seq)]

	names_color = names_seq.loc[names_seq['FundName'].isin(dom), 'Color'].to_list()

	nearest = alt.selection(type='single', nearest=True, on='mouseover',
			fields=['Dates'], empty='none')

	base = alt.Chart(data).mark_line().encode(
			x=alt.X('Dates:T', title=None ,axis=alt.Axis(format = ("%d-%b-%Y"))),
			y = alt.Y(y_col + ':Q', title=None, axis=alt.Axis(format= num_format), scale=alt.Scale(zero=False)),
			color=alt.Color( color_col + ':N', scale=alt.Scale(range= names_color, domain=dom))
		)


	selectors = alt.Chart(data).mark_point().encode(
			x='Dates:T',
			opacity=alt.value(0),
		).add_selection(
			nearest
		)

	points = base.mark_point().encode(
			opacity=alt.condition(nearest, alt.value(1), alt.value(0))
		)

	text = base.mark_text(align='left', dx=-50, dy=10).encode(
			text=alt.condition(nearest, alt.Text(y_col +':Q', format=num_format), alt.value(' ')),
		)

	rules = alt.Chart(data).mark_rule(color='#E7E8F0').encode(
			x = 'Dates:T'
		).transform_filter(
		nearest)

	return alt.layer(
			base, selectors, points, text, rules
			).properties(
			height = 500
			)


@st.cache(allow_output_mutation=True)
def performance_grouped_bar_graph(data, x_col ,y_col, title, dom , min_max, y_label_show = True, x_label_show=False):
	bars = alt.Chart(data).mark_bar().encode(
					x = alt.X(x_col + ':N', title = None, sort="descending", axis=alt.Axis(labels=x_label_show, tickSize=0)),
					y = alt.Y(y_col + ':Q', title = None, axis=alt.Axis(format='%', labels=y_label_show), scale=alt.Scale(domain=min_max)),
					color = alt.Color(x_col + ':N',scale=alt.Scale(domain=dom), legend=None),
					tooltip = alt.Tooltip(y_col + ':Q', format=".2%")
					).properties(
						title = title,
						height = 400
					).configure_title(fontSize=14, orient = 'bottom')

	return bars


def legend_graph(data, col_name, dom):
	return alt.Chart(data).mark_bar().encode(
			color = alt.Color(col_name + ':N', scale=alt.Scale(domain=dom))
			).properties(
			height=35)


@st.cache(allow_output_mutation=True)
def draw_full_holding_graph(data):
	slider_page = alt.binding_range(min=1, max=len(data)/10, step=1, name='Number of holdings (10/page):')
	selector_page = alt.selection_single(name="PageSelector", fields=['page'],
                                    bind=slider_page, init={'page': 1})
	base = alt.Chart(data).transform_calculate(
    			combined=alt.datum.Flag + '    ' + alt.datum.Holding
    		).encode(
    			x = alt.X('Weight:Q', axis = alt.Axis(format='.1%')),
				y = alt.Y('combined:N', sort='-x', axis=alt.Axis(labelAlign='left', labelPadding=180, labelLimit=180))			
    		)
	bars = base.mark_bar(align='left', color='#123FED').encode(
    			tooltip = ['Holding', 'Country', 'Sector', alt.Tooltip('Weight', format = '.2%')]
    		).properties(height=alt.Step(35))

	text = base.mark_text(align='right', color='white').encode(
			text = alt.Text('Sector:N')
			)

	return (bars + text).transform_window(
        		rank = 'rank(Weight)',
    		).add_selection(
        		selector_page
    		).transform_filter(
        		'(datum.rank > (PageSelector.page - 1) * 10) & (datum.rank <= PageSelector.page * 10)'
    		).properties(
    			title = """All portfolio holdings"""
    		)


def fundflow_graph(data, names_pd):
	dates = data.index 
	fig = go.Figure()
	funds = pd.DataFrame(names_pd['FundName'], index=names_pd['ISINCode'], columns=['FundName'])
	colors, _ = color_palette()
	colors_pd = pd.DataFrame(colors[:len(names_pd)], index=names_pd['ISINCode'],columns=['Color'])


	for i, isin in enumerate(data.columns):
		fig.add_trace(go.Bar(x=dates,
						y = data[isin],
						name = isin,
						marker_color = colors_pd.loc[isin, 'Color']
			))
	fig.update_layout(
		margin = dict(l=30, r =30, t=20, b=50),
		height = 500,
		showlegend = False,
		font_family = 'Sans-serif',
		font_size = 14,
		barmode='group',
		bargap=0.15, # gap between bars of adjacent location coordinates.
    	bargroupgap=0.1 # gap between bars of the same location coordinate.
		)
	fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E7E8F0')
	fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E7E8F0')
	fig.layout.plot_bgcolor = 'white'
	return fig

def dividend_graph(data):
	selection = alt.selection_single(
    	fields=['Dates:T'] ,nearest=True, on='mouseover', empty='none', clear='mouseout'
	)

	base = alt.Chart(data).encode(
			x=alt.X('Dates:T', title=None, axis=alt.Axis(format=("%d-%b-%Y"))),	
			#color=alt.Color( color_col + ':N', scale=alt.Scale(range= names_color, domain=dom))
		)

	lines = base.mark_line(color='#304FFE').encode(
			y = alt.Y('Yield:Q', title='Dividend Yield', axis=alt.Axis(format= '.1%')),
		)

	dots = base.mark_circle(size=50, color='#48cae4').encode(
			y = alt.Y('Dividend', title='Dividend', axis=alt.Axis(format= '.1f', labelAlign='left'))
		)

	rules = base.mark_rule().transform_calculate(
    			Dividend=alt.datum.Currency + ' ' + alt.datum.Dividend
    	).encode(
		opacity=alt.condition(selection, alt.value(0.3), alt.value(0)),
    	tooltip=[alt.Tooltip('Dates', type='temporal'), alt.Tooltip('Yield:Q', format='.1%'), 'Dividend:N']
		).add_selection(selection)

	 
	return alt.layer(lines + rules, dots).resolve_scale(y='independent').properties(height=400)
	#return (lines + rules)