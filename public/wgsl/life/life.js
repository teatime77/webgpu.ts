InitCells();
RenderCells();

swapPingPong(CellBuffer);
yield;

while(true){
    UpdateCells();
    RenderCells();

    swapPingPong(CellBuffer);
    yield;
}