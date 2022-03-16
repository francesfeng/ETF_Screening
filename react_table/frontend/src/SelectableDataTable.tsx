import * as React from 'react';
import { useState, useEffect, ReactText } from "react"
import clsx from 'clsx';
import { DataGrid, GridColDef,GridValueGetterParams, GridSelectionModel, GridCellParams, GridValueFormatterParams, GridToolbarContainer, GridToolbarColumnsButton, GridToolbarFilterButton, GridToolbarDensitySelector} from '@mui/x-data-grid';

import {ComponentProps, Streamlit, withStreamlitConnection} from "streamlit-component-lib"
import Link from '@mui/material/Link'

function number_formatter(num) {
  if (Math.abs(num) > 999999999 ) {
    return Math.sign(num)*Number((Math.abs(num)/1000000000).toFixed(1)) + 'B'
  } else if (Math.abs(num) > 999999) {
    return Math.sign(num)*Number((Math.abs(num)/1000000).toFixed(1)) + 'M'
  } else if ( Math.abs(num) > 999) {
    return Math.sign(num)*Number((Math.abs(num)/1000).toFixed(1)) + 'K'
  } else {
    return  Math.sign(num)*Number(Math.abs(num).toFixed(1))
  }
}

function currency_formatter(num, currency) {
  let value;
  const sign = Math.sign(num) === 1 ? '' : '-'
  if (currency ==='USD') {
    value = sign+'$'+number_formatter(Math.abs(Number(num)))
  } else if (currency === 'EUR') {
    value = sign+'€'+number_formatter(Math.abs(Number(num)))
  } else if (currency === 'GBP') {
    value = sign+'£'+number_formatter(Math.abs(Number(num)))
  } else if (currency === 'JPY') {
    value = sign+'¥'+number_formatter(Math.abs(Number(num)))
  } else if (currency === 'GBX') {
    value = sign + 'GBp ' + number_formatter(Math.abs(Number(num)))
  } else {
    value = number_formatter(Math.abs(Number(num)))
  }
  return value
}


function currency_reverse(curr) {
  let originalCurrency
  if (curr === '$') {
    originalCurrency = 'USD'
  } else if (curr === '€') {
    originalCurrency = 'EUR'
  } else if (curr === '£') {
    originalCurrency = 'GBP'
  } else if (curr === '¥') {
    originalCurrency = 'JPY'
  } else {
    originalCurrency = curr
  }
  return originalCurrency
}

function rank_formatter(rank) {
  let rank_stars
  if (rank===5) {
    rank_stars = '⭐⭐⭐⭐⭐'
  } else if(rank===4) {
    rank_stars = '⭐⭐⭐⭐'
  } else if(rank===3) {
    rank_stars = '⭐⭐⭐'
  } else if(rank===2) {
    rank_stars = '⭐⭐'
  } else {
    rank_stars = '⭐'
  }
           
  return rank_stars
}


function CustomToolbar() {
  return (
    <GridToolbarContainer>
      <GridToolbarColumnsButton />
      <GridToolbarFilterButton />
      <GridToolbarDensitySelector />
    </GridToolbarContainer>
  );
}


const digits_formatter = (params: GridValueFormatterParams) => {
        return `${params.value ? Number(params.value).toFixed(2) + '' : ''}`
      }

const number_color_formatter = (params: GridCellParams<number>) =>
        clsx('super-app', {
          negative: params.value < 0,
          positive: params.value > 0,
        })

const str_color_formatter = (params: GridCellParams<number>) =>
        clsx('super-app', {
          // negative: params.value < 0,
          // positive: params.value > 0,
          negative: String(params.value).substring(0,1) === '-',
          positive: String(params.value).substring(0,1) !== '0',
        })


function calender_year_columns(data, column_names) {
  let columns_calender = []
  column_names.forEach(item => {
    if (item === 'id' || item === 'ISINCode' || item === 'Exchange') {

    } else if (item === 'ExchangeTicker') {
      columns_calender.push({
        field: 'ExchangeTicker',
        headerName: 'Ticker',
        width: 100,
      })
    } else if (item === 'FundName') {
      columns_calender.push({
        field: 'FundName',
        headerName: 'Name',
        width: 300,
      })
    } else if (item === 'YTD') {
      columns_calender.push({
        field: 'YTD',
        headerName: 'YTD (%)',
        type: 'number',
        width: 100,
      })
    } else {
      columns_calender.push({
        field: item,
        headerName: item.substr(4,4) + ' (%)',
        type: 'number',
        minWidth: 100,
        flex: 1,
        valueFormatter: digits_formatter,
        cellClassName: number_color_formatter,
      })     
    }
  })
  return columns_calender
}

const SelectableDataTable: React.FC<ComponentProps> = props => {
  useEffect(() => {
    Streamlit.setFrameHeight()
  })


  const handleSelectionChange = (value: ReactText[]): void => {
      setSelectionModel(value)
      Streamlit.setComponentValue(value)
    }

  const [selectionModel, setSelectionModel] =  useState<GridSelectionModel>([]);

  const display = props.args.display;
  const currency = props.args.currency;
  const return_type = props.args.return_type;

  const columns_overview: GridColDef[] = [
    {
      field: 'ExchangeTicker',
      headerName: 'Ticker',
      width: 100,
      hideable: false,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'FundName',
      headerName: 'Name',
      width: 300,
      hideable: false,
      renderCell: (params) => {
        return (<Link href="${params.row.FundName}" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'Exchange',
      headerName: 'Exchange',
      width: 110,
    },
    {
      field: 'Rank5',
      headerName: 'Rank',
      type: 'number',
      width: 100,
      align: 'left',
      headerAlign: 'center',
      valueFormatter: (params: GridValueFormatterParams) =>{
        return `${rank_formatter(Number(params.value))}`
      }
    },
    {
      field: 'Price',
      headerName: 'Price',
      description: 'The price is as of June 19, 2021',
      sortable: false,
      width: 100,
      valueGetter: (params: GridValueGetterParams) =>
        `${params.row.TradeCurrency || ''} ${params.row.Price || ' '} `,
    },
    {
      field: 'Volume3M',
      headerName: 'Volume',
      type: 'number',
      description: '3 Months Average Daily Volume (in shares)',
      width: 120,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${number_formatter(Number(params.value as number))}`;
      },
    },
    {
      field: 'FundCurrency',
      headerName: 'Fund Currency',
      description: 'The price is as of June 19, 2021',
      width: 130,
      valueGetter: (params: GridValueGetterParams) => {
        return `${currency_reverse(params.value)}`
      }
    },
    {
      field: 'NAV',
      headerName: 'NAV',
      description: 'Net Asset Value (as of June 2019, 2021',
      sortable: false,
      width: 100,
      valueGetter: (params: GridValueGetterParams) =>
        `${params.row.FundCurrency || ''} ${Number(params.row.NAV).toFixed(2) || ' '} `,
    },
    {
      field: 'NAV_1DChange',
      headerName: 'Daily Change',
      type: 'number',
      description: 'The price is as of June 19, 2021',
      width: 120,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${Number(params.value).toFixed(2)}%`
      },
      cellClassName: (params: GridCellParams<number>) =>
        clsx('super-app', {
          negative: params.value < 0,
          positive: params.value > 0,
        }),
    },
    {
      field: 'AUM',
      headerName: 'AUM',
      description: 'Asset Under Management (as of June 2019, 2021)',
      width: 120,
      type: 'number',

      valueGetter: (params: GridValueGetterParams) => {
        return currency === 'Fund currency' ? `${params.row.FundCurrency || ''}${currency_formatter(params.row.AUM, currency)}` : `${currency_formatter(params.row.AUM, currency)}`
      }
    },
    {
      field: 'DistributionIndicator',
      headerName: 'Distribution',
      width: 110,
    },
    {
      field: 'TotalExpenseRatio',
      headerName: 'Fees (%)',
      type: 'number',
      width: 90,
    },
    {
      field: 'ETPIssuerName',
      headerName: 'Issuer',
      width: 110,
    },
    {
      field: 'IndexName',
      headerName: 'Index',
      width: 200,
    },
  ];


  const columns_cumulative: GridColDef[] = [
    {
      field: 'ExchangeTicker',
      headerName: 'Ticker',
      width: 100,
      hideable: false,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'FundName',
      headerName: 'Name',
      width: 300,
      hideable: false,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: '1M',
      headerName: '1M (%)',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '3M',
      headerName: '3M (%)',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '6M',
      headerName: '6M (%)',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: 'YTD',
      headerName: 'YTD (%)',
      type: 'number',
      minWidth: 90,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '1Y',
      headerName: '1Y (%)',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '3Y',
      headerName: '3Y (%)',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '5Y',
      headerName: '5Y (%)',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '10Y',
      headerName: '10Y (%)',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    ]

  const columns_annualised: GridColDef[] = [
    {
      field: 'ExchangeTicker',
      headerName: 'Ticker',
      width: 100,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'FundName',
      headerName: 'Name',
      width: 300,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: '1Y',
      headerName: '1Y (%)',
      type: 'number',
      minWidth: 90,
      flex: 1,
      headerAlign: 'center',
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '3Y',
      headerName: '3Y (%)',
      type: 'number',
      minWidth: 90,
      flex: 1,
      headerAlign: 'center',
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '5Y',
      headerName: '5Y (%)',
      type: 'number',
      minWidth: 90,
      flex: 1,
      headerAlign: 'center',
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    {
      field: '10Y',
      headerName: '10Y (%)',
      type: 'number',
      minWidth: 90,
      flex: 1,
      headerAlign: 'center',
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    ]


  const columns_flow: GridColDef[] = [
    {
      field: 'ExchangeTicker',
      headerName: 'Ticker',
      width: 100,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'FundName',
      headerName: 'Name',
      width: 300,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'Currency',
      headerName: 'Fund Currency',
      width: 130,
      align: 'center'
    },
    {
      field: 'AUM',
      headerName: 'AUM',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
    },
    {
      field: '1M',
      headerName: '1M',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
      cellClassName: number_color_formatter,
    },
    {
      field: '3M',
      headerName: '3M',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
      cellClassName: number_color_formatter,
    },
    {
      field: '6M',
      headerName: '6M',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
      cellClassName: number_color_formatter,
    },
    {
      field: 'YTD',
      headerName: 'YTD',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
      cellClassName: number_color_formatter,
    },
    {
      field: '1Y',
      headerName: '1Y',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
      cellClassName: number_color_formatter,
    },
    {
      field: '3Y',
      headerName: '3Y',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
      cellClassName: number_color_formatter,
    },
    {
      field: '5Y',
      headerName: '5Y',
      type: 'number',
      minWidth: 80,
      flex: 1,
      valueFormatter: (params: GridValueFormatterParams) => {
        return `${currency_formatter(params.value, currency)}`;
      },
      cellClassName: number_color_formatter,
    },
    ]

  const columns_div: GridColDef[] = [
    {
      field: 'ExchangeTicker',
      headerName: 'Ticker',
      width: 100,
      hideable: false,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'FundName',
      headerName: 'Name',
      width: 300,
      hideable: false,
      renderCell: (params) => {
        return (<Link href="#" underline="hover" color="inherit">{params.value}</Link>)
      }
    },
    {
      field: 'DistributionIndicator',
      headerName: 'Dividend Treatment',
      width: 180,
      hideable: false
    },
    {
      field: 'CashFlowFrequency',
      headerName: 'Dividend Frequency',
      width: 180,
      hideable: false
    },
    {
      field: 'exDivDate',
      headerName: 'ex-Dividend Date',
      width: 150,
      hideable: false
    },
    {
      field: 'Dividend',
      headerName: 'Last Dividend',
      description: 'Asset Under Management (as of June 2019, 2021)',
      width: 120,
      type: 'number',
      valueGetter: (params: GridValueGetterParams) => {
        return `${currency_formatter(params.row.Dividend, params.row.Currency) || ''}`
      }
    },
    {
      field: 'Yield',
      headerName: 'Yield (%)',
      type: 'number',
      minWidth: 100,
      flex: 1,
      valueFormatter: digits_formatter,
    },
    {
      field: 'DivGrowth',
      headerName: 'Div Growth',
      type: 'number',
      minWidth: 100,
      flex: 1,
      valueGetter: (params: GridValueGetterParams) => {
        return `${currency_formatter(params.row.Dividend, params.row.Currency) || ''}`
      },
      // valueFormatter: digits_formatter,
      cellClassName: str_color_formatter,
    },
    {
      field: 'DivGrowthPct',
      headerName: 'Div Growth (%)',
      type: 'number',
      minWidth: 130,
      flex: 1,
      valueFormatter: digits_formatter,
      cellClassName: number_color_formatter,
    },
    ]



  const rows = props.args.data;
  let columns : GridColDef[] = []
  let fieldHeaders = []

  if (display === 'Overview')  {
    columns = columns_overview
    } else if (display === 'Performance') {
        if (return_type === 'Cumulative') {
        columns = columns_cumulative
      } else if (return_type === 'Annualised') {
          columns = columns_annualised
      } else {
        fieldHeaders = Object.keys(rows[0])
        columns = calender_year_columns(rows, fieldHeaders)
      }  
    } else if (display === 'Fund Flow') {
      columns = columns_flow
    } else if (display === 'Income'){
      columns = columns_div
    }else {

    }
    

  const columnsHide = (headers) => {
    interface hideColumnsType {
      [key: string]: boolean
    }
    let yearsColumnsHide: hideColumnsType = {}
    
    if (headers.length > 12) {
      headers.slice(12).forEach(item => {
        yearsColumnsHide[item] = false
      })
    }
    yearsColumnsHide['Price'] = false
    yearsColumnsHide['Volume3M'] = false
  
    //console.log(yearsColumnsHide)
    return yearsColumnsHide
  }
    
  return (
     <div style={{ height: 700, width: '100%' }}>
     <DataGrid
        rows={rows}
        columns={columns}
        pageSize={10}
        rowsPerPageOptions={[10]}
        checkboxSelection
        //disableSelectionOnClick

        onSelectionModelChange={(newSelectionModel) => {
           handleSelectionChange(newSelectionModel);
         }}
        selectionModel={selectionModel}
        //initialState={{ pinnedColumns: { left: ['Ticker'] } }} need mui/x-data-grid-pro commercial license
        
        components={{
            Toolbar: CustomToolbar,
          }}
        sx={{
          '& .super-app.negative': {
            color: '#D81414',
            fontWeight: 'bold',
          },
          '& .super-app.positive': {
            color: '#0DAA5B',
            fontWeight: 'bold',
          },
        }}
        initialState={{
          columns: {
            columnVisibilityModel: {
              ...columnsHide(fieldHeaders),
              Currency: display === 'Fund Flow' && currency === 'Fund currency' ? true : false
            }
        },
      }}
      />

    </div>
        
  )
}

export default withStreamlitConnection(SelectableDataTable)
