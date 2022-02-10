import * as React from 'react';
import { useState, useEffect, ReactText } from "react"
import { DataGrid, GridColDef, GridValueGetterParams, GridSelectionModel} from '@mui/x-data-grid';

import {ComponentProps, Streamlit, withStreamlitConnection} from "streamlit-component-lib"

const columns: GridColDef[] = [
  { field: 'id', headerName: 'ID', width: 90 },
  {
    field: 'firstName',
    headerName: 'First name',
    width: 150,
    editable: true,
  },
  {
    field: 'lastName',
    headerName: 'Last name',
    width: 150,
    editable: true,
  },
  {
    field: 'age',
    headerName: 'Age',
    type: 'number',
    width: 110,
    editable: true,
  },
  {
    field: 'fullName',
    headerName: 'Full name',
    description: 'This column has a value getter and is not sortable.',
    sortable: false,
    width: 160,
    valueGetter: (params: GridValueGetterParams) =>
      `${params.row.firstName || ''} ${params.row.lastName || ''}`,
  },
];



const SelectableDataTable: React.FC<ComponentProps> = props => {
  useEffect(() => {
    Streamlit.setFrameHeight()
  })

   const handleSelectionChange = (value: ReactText[]): void => {
      setSelectionModel(value)
      Streamlit.setComponentValue(value)
    }

  const [selectionModel, setSelectionModel] =  useState<GridSelectionModel>([]);

  const rows = props.args.data
  
  return (
     <div style={{ height: 400, width: '100%' }}>
      <DataGrid
        rows={rows}
        columns={columns}
        pageSize={5}
        rowsPerPageOptions={[5]}
        checkboxSelection
        //disableSelectionOnClick
        onSelectionModelChange={(newSelectionModel) => {
           handleSelectionChange(newSelectionModel);
         }}
        selectionModel={selectionModel}
      />

    </div>
      
  )
}

export default withStreamlitConnection(SelectableDataTable)
