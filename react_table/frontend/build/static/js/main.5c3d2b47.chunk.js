(this["webpackJsonpselectable-dataframe-example"]=this["webpackJsonpselectable-dataframe-example"]||[]).push([[0],{149:function(e,r,a){e.exports=a(164)},164:function(e,r,a){"use strict";a.r(r);var t=a(0),n=a.n(t),i=a(54),l=a.n(i),u=a(25),d=a(12),c=a(11),m=a(70),o=a(92),h=a(235);function s(e){return Math.abs(e)>999999999?Math.sign(e)*Number((Math.abs(e)/1e9).toFixed(1))+"B":Math.abs(e)>999999?Math.sign(e)*Number((Math.abs(e)/1e6).toFixed(1))+"M":Math.abs(e)>999?Math.sign(e)*Number((Math.abs(e)/1e3).toFixed(1))+"K":Math.sign(e)*Number(Math.abs(e).toFixed(1))}function f(e,r){var a=1===Math.sign(e)?"":"-";return"USD"===r?a+"$"+s(Math.abs(Number(e))):"EUR"===r?a+"\u20ac"+s(Math.abs(Number(e))):"GBP"===r?a+"\xa3"+s(Math.abs(Number(e))):"JPY"===r?a+"\xa5"+s(Math.abs(Number(e))):"GBX"===r?a+"GBp "+s(Math.abs(Number(e))):s(Math.abs(Number(e)))}function N(){return t.createElement(m.c,null,t.createElement(m.b,null),t.createElement(m.e,null),t.createElement(m.d,null))}var v=function(e){return"".concat(e.value?Number(e.value).toFixed(2)+"":"")},b=function(e){return Object(c.a)("super-app",{negative:e.value<0,positive:e.value>0})},p=function(e){return Object(c.a)("super-app",{negative:"-"===String(e.value).substring(0,1),positive:"0"!==String(e.value).substring(0,1)})};var F=Object(o.b)((function(e){Object(t.useEffect)((function(){o.a.setFrameHeight()}));var r=Object(t.useState)([]),a=Object(d.a)(r,2),n=a[0],i=a[1],l=e.args.display,F=e.args.currency,w=e.args.return_type,y=[{field:"ExchangeTicker",headerName:"Ticker",width:100,hideable:!1,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"FundName",headerName:"Name",width:300,hideable:!1,renderCell:function(e){return t.createElement(h.a,{href:"${params.row.FundName}",underline:"hover",color:"inherit"},e.value)}},{field:"Exchange",headerName:"Exchange",width:110},{field:"Rank5",headerName:"Rank",type:"number",width:100,align:"left",headerAlign:"center",valueFormatter:function(e){return"".concat(5===(r=Number(e.value))?"\u2b50\u2b50\u2b50\u2b50\u2b50":4===r?"\u2b50\u2b50\u2b50\u2b50":3===r?"\u2b50\u2b50\u2b50":2===r?"\u2b50\u2b50":"\u2b50");var r}},{field:"Price",headerName:"Price",description:"The price is as of June 19, 2021",sortable:!1,width:100,valueGetter:function(e){return"".concat(e.row.TradeCurrency||""," ").concat(e.row.Price||" "," ")}},{field:"Volume3M",headerName:"Volume",type:"number",description:"3 Months Average Daily Volume (in shares)",width:120,valueFormatter:function(e){return"".concat(s(Number(e.value)))}},{field:"FundCurrency",headerName:"Fund Currency",description:"The price is as of June 19, 2021",width:130,valueGetter:function(e){return"".concat("$"===(r=e.value)?"USD":"\u20ac"===r?"EUR":"\xa3"===r?"GBP":"\xa5"===r?"JPY":r);var r}},{field:"NAV",headerName:"NAV",description:"Net Asset Value (as of June 2019, 2021",sortable:!1,width:100,valueGetter:function(e){return"".concat(e.row.FundCurrency||""," ").concat(Number(e.row.NAV).toFixed(2)||" "," ")}},{field:"NAV_1DChange",headerName:"Daily Change",type:"number",description:"The price is as of June 19, 2021",width:120,valueFormatter:function(e){return"".concat(Number(e.value).toFixed(2),"%")},cellClassName:function(e){return Object(c.a)("super-app",{negative:e.value<0,positive:e.value>0})}},{field:"AUM",headerName:"AUM",description:"Asset Under Management (as of June 2019, 2021)",width:120,type:"number",valueGetter:function(e){return"Fund currency"===F?"".concat(e.row.FundCurrency||"").concat(f(e.row.AUM,F)):"".concat(f(e.row.AUM,F))}},{field:"DistributionIndicator",headerName:"Distribution",width:110},{field:"TotalExpenseRatio",headerName:"Fees (%)",type:"number",width:90},{field:"ETPIssuerName",headerName:"Issuer",width:110},{field:"IndexName",headerName:"Index",width:200}],x=[{field:"ExchangeTicker",headerName:"Ticker",width:100,hideable:!1,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"FundName",headerName:"Name",width:300,hideable:!1,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"1M",headerName:"1M (%)",type:"number",minWidth:80,flex:1,valueFormatter:v,cellClassName:b},{field:"3M",headerName:"3M (%)",type:"number",minWidth:80,flex:1,valueFormatter:v,cellClassName:b},{field:"6M",headerName:"6M (%)",type:"number",minWidth:80,flex:1,valueFormatter:v,cellClassName:b},{field:"YTD",headerName:"YTD (%)",type:"number",minWidth:90,flex:1,valueFormatter:v,cellClassName:b},{field:"1Y",headerName:"1Y (%)",type:"number",minWidth:80,flex:1,valueFormatter:v,cellClassName:b},{field:"3Y",headerName:"3Y (%)",type:"number",minWidth:80,flex:1,valueFormatter:v,cellClassName:b},{field:"5Y",headerName:"5Y (%)",type:"number",minWidth:80,flex:1,valueFormatter:v,cellClassName:b},{field:"10Y",headerName:"10Y (%)",type:"number",minWidth:80,flex:1,valueFormatter:v,cellClassName:b}],C=[{field:"ExchangeTicker",headerName:"Ticker",width:100,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"FundName",headerName:"Name",width:300,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"1Y",headerName:"1Y (%)",type:"number",minWidth:90,flex:1,headerAlign:"center",valueFormatter:v,cellClassName:b},{field:"3Y",headerName:"3Y (%)",type:"number",minWidth:90,flex:1,headerAlign:"center",valueFormatter:v,cellClassName:b},{field:"5Y",headerName:"5Y (%)",type:"number",minWidth:90,flex:1,headerAlign:"center",valueFormatter:v,cellClassName:b},{field:"10Y",headerName:"10Y (%)",type:"number",minWidth:90,flex:1,headerAlign:"center",valueFormatter:v,cellClassName:b}],g=[{field:"ExchangeTicker",headerName:"Ticker",width:100,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"FundName",headerName:"Name",width:300,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"Currency",headerName:"Fund Currency",width:130,align:"center"},{field:"AUM",headerName:"AUM",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))}},{field:"1M",headerName:"1M",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))},cellClassName:b},{field:"3M",headerName:"3M",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))},cellClassName:b},{field:"6M",headerName:"6M",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))},cellClassName:b},{field:"YTD",headerName:"YTD",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))},cellClassName:b},{field:"1Y",headerName:"1Y",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))},cellClassName:b},{field:"3Y",headerName:"3Y",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))},cellClassName:b},{field:"5Y",headerName:"5Y",type:"number",minWidth:80,flex:1,valueFormatter:function(e){return"".concat(f(e.value,F))},cellClassName:b}],M=[{field:"ExchangeTicker",headerName:"Ticker",width:100,hideable:!1,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"FundName",headerName:"Name",width:300,hideable:!1,renderCell:function(e){return t.createElement(h.a,{href:"#",underline:"hover",color:"inherit"},e.value)}},{field:"DistributionIndicator",headerName:"Dividend Treatment",width:180,hideable:!1},{field:"CashFlowFrequency",headerName:"Dividend Frequency",width:180,hideable:!1},{field:"exDivDate",headerName:"ex-Dividend Date",width:150,hideable:!1},{field:"Dividend",headerName:"Last Dividend",description:"Asset Under Management (as of June 2019, 2021)",width:120,type:"number",valueGetter:function(e){return"".concat(f(e.row.Dividend,e.row.Currency)||"")}},{field:"Yield",headerName:"Yield (%)",type:"number",minWidth:100,flex:1,valueFormatter:v},{field:"DivGrowth",headerName:"Div Growth",type:"number",minWidth:100,flex:1,valueGetter:function(e){return"".concat(f(e.row.Dividend,e.row.Currency)||"")},cellClassName:p},{field:"DivGrowthPct",headerName:"Div Growth (%)",type:"number",minWidth:130,flex:1,valueFormatter:v,cellClassName:b}],E=e.args.data,Y=[],D=[];"Overview"===l?Y=y:"Performance"===l?Y="Cumulative"===w?x:"Annualised"===w?C:function(e,r){var a=[];return r.forEach((function(e){"id"===e||"ISINCode"===e||"Exchange"===e||("ExchangeTicker"===e?a.push({field:"ExchangeTicker",headerName:"Ticker",width:100}):"FundName"===e?a.push({field:"FundName",headerName:"Name",width:300}):"YTD"===e?a.push({field:"YTD",headerName:"YTD (%)",type:"number",width:100}):a.push({field:e,headerName:e.substr(4,4)+" (%)",type:"number",minWidth:100,flex:1,valueFormatter:v,cellClassName:b}))})),a}(0,D=Object.keys(E[0])):"Fund Flow"===l?Y=g:"Income"===l&&(Y=M);return t.createElement("div",{style:{height:700,width:"100%"}},t.createElement(m.a,{rows:E,columns:Y,pageSize:10,rowsPerPageOptions:[10],checkboxSelection:!0,onSelectionModelChange:function(e){var r;i(r=e),o.a.setComponentValue(r)},selectionModel:n,components:{Toolbar:N},sx:{"& .super-app.negative":{color:"#D81414",fontWeight:"bold"},"& .super-app.positive":{color:"#0DAA5B",fontWeight:"bold"}},initialState:{columns:{columnVisibilityModel:Object(u.a)(Object(u.a)({},function(e){var r={};return e.length>12&&e.slice(12).forEach((function(e){r[e]=!1})),r.Price=!1,r.Volume3M=!1,r}(D)),{},{Currency:"Fund Flow"===l&&"Fund currency"===F})}}}))}));l.a.render(n.a.createElement(F,null),document.getElementById("root"))}},[[149,1,2]]]);
//# sourceMappingURL=main.5c3d2b47.chunk.js.map