namespace webgpu_ts {
//

let iii : number = 0;
let last_false : number;
const idxs = new Set<number>();

function shuffleArray(array : Float32Array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]]; // Swap elements
    }
    return array;
}

export function bitonic_sort_test(){
    let max_last_false = 0;
    for(const c of range(400)){
        iii = 0;
        last_false = 0;
        const x = new Float32Array(8).map(x => Math.round(Math.random() * 1000));
        shuffleArray(x);
        bitonic_sort(0, 0, true, x, 0, x.length);
        for(let i = 0; i < x.length - 1; i++){
            assert(x[i] <= x[i + 1]);
        }
        // msg(`${x}`);
        max_last_false = Math.max(max_last_false, last_false);
        msg(`last false:${max_last_false} idxs:${Array.from(idxs).sort((a,b) => a - b)}`);
        msg("");
    }

}

export function isSorted(up: boolean, x : Float32Array, offset : number, count : number) : boolean {
    if(up){
        return range(count).every(i => i == 0 || x[offset + i - 1] <= x[offset + i]);
    }
    else{
        return range(count).every(i => i == 0 || x[offset + i - 1] >= x[offset + i]);
    }
}

function splited(up: boolean, x: Float32Array){
    const half = x.length / 2;
    assert(x.slice(0, half).length == half && x.slice(half).length == half);
    if(up){
        const A_max = Math.max(...x.slice(0, half));
        const B_min = Math.min(...x.slice(half));
        return A_max <= B_min
    }
    else{
        const A_min = Math.min(...x.slice(0, half));
        const B_max = Math.max(...x.slice(half));
        return A_min >= B_max;
    }
}

function bitonic_seq(x : Float32Array){
    if(x.length == 1){
        return true;
    }

    const half = x.length / 2;
    return isSorted(true, x, 0, half) && isSorted(false, x, half, half);

}

function sp(nest : number){
    return " ".repeat(nest * 4);
}

export function bitonic_sort(pos:number, nest:number, up: boolean, x: Float32Array, offset : number, count : number){
    if(count <= 1){
        return;
    }

    const half = count / 2;

    bitonic_sort(1, nest + 1, true , x, offset, half);

    bitonic_sort(2, nest + 1, false, x, offset + half, half);

    bitonic_merge(1, nest + 1, up, x, offset, count);

    iii++;
    assert( splited(up, x.slice(offset, offset + count)) );
    msg(`${iii} sort :${sp(nest)} pos:${pos} offset:${offset} count:${count} ${up?'↑':'↓'}`);
}

function bitonic_merge(pos:number, nest:number, up: boolean, x : Float32Array, offset : number, count : number){
    //  入力xがバイトニック列であれば、並べ替えられて出力される
    if(count == 1){
        return;
    }

    const ok = bitonic_seq( x.slice(offset, offset + count) );
    if(!ok){
        msg(`not bitonic:${x.slice(offset, offset + count)}`);
    }
    bitonic_compare(up, x, offset, count);
    assert(splited(up, x.slice(offset, offset + count)));

    const half = count / 2;
    bitonic_merge(2, nest + 1, up, x, offset, half);

    bitonic_merge(3, nest + 1, up, x, offset + half, half);

    iii++;
    if(!ok){
        last_false = iii;
        idxs.add(iii);
    }
    assert(isSorted(up, x, offset, count) && splited(up, x.slice(offset, offset + count)));

    msg(`${iii} merge:${sp(nest)} ${pos} offset:${offset} count:${count} ${up?'↑':'↓'} ${ok}`);
}

function bitonic_compare(up: boolean, x : Float32Array, offset : number, count : number){
    const dist = count / 2;
    let swapped = false;
    for(let i = 0; i < dist; i++){
        const i1 = offset + i;
        const i2 = offset + dist + i;
        // if(up && swapped){
        //     assert(up == (x[i1] > x[i2]));
        // }
        if(up == (x[i1] > x[i2])){
            [x[i1], x[i2]] = [x[i2], x[i1]];
            swapped = true;
        }
    }
}

}