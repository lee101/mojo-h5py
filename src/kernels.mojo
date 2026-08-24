"""HDF5 raw-chunk filters exposed through a small C ABI."""

from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U16Ptr = UnsafePointer[UInt16, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def shuffle8_range(
    src_addr: Int, dst_addr: Int, elements: Int, begin: Int, end: Int
):
    comptime W = simdwidthof[DType.float64]()
    var src_words = U64Ptr(unsafe_from_address=src_addr)
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var element = begin
    while element + W <= end:
        var words = src_words.load[width=W, alignment=1](element)
        for byte_index in range(8):
            var shift = SIMD[DType.uint64, W](8 * byte_index)
            var values = (words >> shift).cast[DType.uint8]()
            dst.store[alignment=1](byte_index * elements + element, values)
        element += W
    while element < end:
        for byte_index in range(8):
            dst[byte_index * elements + element] = src[8 * element + byte_index]
        element += 1


def unshuffle8_range(
    src_addr: Int, dst_addr: Int, elements: Int, begin: Int, end: Int
):
    comptime W = simdwidthof[DType.float64]()
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var dst_words = U64Ptr(unsafe_from_address=dst_addr)
    var element = begin
    while element + W <= end:
        var words = SIMD[DType.uint64, W](0)
        for byte_index in range(8):
            var values = src.load[width=W, alignment=1](
                byte_index * elements + element
            ).cast[DType.uint64]()
            var shift = SIMD[DType.uint64, W](8 * byte_index)
            words |= values << shift
        dst_words.store[alignment=1](element, words)
        element += W
    while element < end:
        for byte_index in range(8):
            dst[8 * element + byte_index] = src[byte_index * elements + element]
        element += 1


def shuffle_bytes(src: BPtr, dst: BPtr, nbytes: Int, element_size: Int):
    if element_size <= 1:
        comptime W = simdwidthof[DType.float64]()
        var i = 0
        while i + W <= nbytes:
            var values = src.load[width=W, alignment=1](i)
            dst.store[alignment=1](i, values)
            i += W
        while i < nbytes:
            dst[i] = src[i]
            i += 1
        return
    if element_size == 8:
        var elements = nbytes // 8
        shuffle8_range(Int(src), Int(dst), elements, 0, elements)
        return
    var elements = nbytes // element_size
    for byte_index in range(element_size):
        var dst_base = byte_index * elements
        for element in range(elements):
            dst[dst_base + element] = src[element * element_size + byte_index]


def unshuffle_bytes(src: BPtr, dst: BPtr, nbytes: Int, element_size: Int):
    if element_size <= 1:
        comptime W = simdwidthof[DType.float64]()
        var i = 0
        while i + W <= nbytes:
            var values = src.load[width=W, alignment=1](i)
            dst.store[alignment=1](i, values)
            i += W
        while i < nbytes:
            dst[i] = src[i]
            i += 1
        return
    if element_size == 8:
        var elements = nbytes // 8
        unshuffle8_range(Int(src), Int(dst), elements, 0, elements)
        return
    var elements = nbytes // element_size
    for byte_index in range(element_size):
        var src_base = byte_index * elements
        for element in range(elements):
            dst[element * element_size + byte_index] = src[src_base + element]


def fletcher32(src: BPtr, nbytes: Int) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var src_words = U16Ptr(unsafe_from_address=Int(src))
    var sum1 = 0
    var sum2 = 0
    var words = (nbytes + 1) // 2
    var full_words = nbytes // 2
    var word_index = 0
    while word_index < words:
        var block = min(360, words - word_index)
        var block_end = word_index + block
        var vector_end = min(block_end, full_words)
        var block_sum1 = 0
        var block_sum2 = 0
        var i = word_index
        while i + W <= vector_end:
            var values = src_words.load[width=W, alignment=1](i).cast[
                DType.int64
            ]()
            var weights = SIMD[DType.int64, W](0)
            comptime for lane in range(W):
                weights[lane] = Int64(block_end - i - lane)
            block_sum1 += Int(values.reduce_add())
            block_sum2 += Int((values * weights).reduce_add())
            i += W
        while i < block_end:
            var byte_index = 2 * i
            var word = Int(src[byte_index])
            if byte_index + 1 < nbytes:
                word |= Int(src[byte_index + 1]) << 8
            block_sum1 += word
            block_sum2 += (block_end - i) * word
            i += 1
        sum2 += block * sum1 + block_sum2
        sum1 += block_sum1
        sum1 = (sum1 & 65535) + (sum1 >> 16)
        sum2 = (sum2 & 65535) + (sum2 >> 16)
        word_index = block_end
    sum1 = (sum1 & 65535) + (sum1 >> 16)
    sum2 = (sum2 & 65535) + (sum2 >> 16)
    return ((sum1 & 65535) << 16) | (sum2 & 65535)


def shuffle8_fletcher32(src: BPtr, dst: BPtr, nbytes: Int):
    comptime W = simdwidthof[DType.float64]()
    var elements = nbytes // 8
    if elements % 2 != 0 or nbytes > 16 * 1024 * 1024:
        shuffle8_range(Int(src), Int(dst), elements, 0, elements)
        var checksum = fletcher32(dst, nbytes)
        dst[nbytes] = UInt8((checksum >> 24) & 255)
        dst[nbytes + 1] = UInt8((checksum >> 16) & 255)
        dst[nbytes + 2] = UInt8((checksum >> 8) & 255)
        dst[nbytes + 3] = UInt8(checksum & 255)
        return

    var factors = SIMD[DType.int64, W](0)
    comptime for lane in range(W):
        if lane % 2 == 0:
            factors[lane] = 1
        else:
            factors[lane] = 256

    var src_words = U64Ptr(unsafe_from_address=Int(src))
    var vector_sum1 = SIMD[DType.int64, W](0)
    var vector_sum2 = SIMD[DType.int64, W](0)
    var word_count = nbytes // 2
    var element = 0
    while element + W <= elements:
        var input_words = src_words.load[width=W, alignment=1](element)
        for byte_index in range(8):
            var shift = SIMD[DType.uint64, W](8 * byte_index)
            var output_bytes = (input_words >> shift).cast[DType.uint8]()
            dst.store[alignment=1](byte_index * elements + element, output_bytes)
            var contributions = output_bytes.cast[DType.int64]() * factors
            var weights = SIMD[DType.int64, W](0)
            comptime for lane in range(W):
                weights[lane] = Int64(
                    word_count
                    - byte_index * (elements // 2)
                    - element // 2
                    - lane // 2
                )
            vector_sum1 += contributions
            vector_sum2 += contributions * weights
        element += W

    var sum1 = Int(vector_sum1.reduce_add())
    var sum2 = Int(vector_sum2.reduce_add())
    while element < elements:
        for byte_index in range(8):
            var output_index = byte_index * elements + element
            var value = src[8 * element + byte_index]
            dst[output_index] = value
            var contribution = Int(value)
            if output_index % 2 != 0:
                contribution <<= 8
            sum1 += contribution
            sum2 += (word_count - output_index // 2) * contribution
        element += 1

    while sum1 >> 16 != 0:
        sum1 = (sum1 & 65535) + (sum1 >> 16)
    while sum2 >> 16 != 0:
        sum2 = (sum2 & 65535) + (sum2 >> 16)
    var checksum = ((sum1 & 65535) << 16) | (sum2 & 65535)
    dst[nbytes] = UInt8((checksum >> 24) & 255)
    dst[nbytes + 1] = UInt8((checksum >> 16) & 255)
    dst[nbytes + 2] = UInt8((checksum >> 8) & 255)
    dst[nbytes + 3] = UInt8(checksum & 255)


@export("mh5_shuffle")
def mh5_shuffle(src_addr: Int, dst_addr: Int, nbytes: Int, element_size: Int) abi("C"):
    shuffle_bytes(bp(src_addr), bp(dst_addr), nbytes, element_size)


@export("mh5_unshuffle")
def mh5_unshuffle(src_addr: Int, dst_addr: Int, nbytes: Int, element_size: Int) abi("C"):
    unshuffle_bytes(bp(src_addr), bp(dst_addr), nbytes, element_size)


@export("mh5_fletcher32")
def mh5_fletcher32(src_addr: Int, nbytes: Int) abi("C") -> Int:
    return fletcher32(bp(src_addr), nbytes)


@export("mh5_shuffle8_fletcher32")
def mh5_shuffle8_fletcher32(src_addr: Int, dst_addr: Int, nbytes: Int) abi("C"):
    shuffle8_fletcher32(bp(src_addr), bp(dst_addr), nbytes)


@export("mh5_copy_bytes")
def mh5_copy_bytes(src_addr: Int, dst_addr: Int, nbytes: Int) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= nbytes:
        var values = src.load[width=W, alignment=1](i)
        dst.store[alignment=1](i, values)
        i += W
    while i < nbytes:
        dst[i] = src[i]
        i += 1
