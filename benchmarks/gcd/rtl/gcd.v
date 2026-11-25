// Small synthesizable GCD benchmark. Technology mapping is supplied by the
// Linux OpenROAD deployment; this file contains no physical-design metrics.
module gcd #(
    parameter WIDTH = 32
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    output reg              busy,
    output reg              done,
    output reg [WIDTH-1:0]  result
);
    reg [WIDTH-1:0] x;
    reg [WIDTH-1:0] y;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy   <= 1'b0;
            done   <= 1'b0;
            result <= {WIDTH{1'b0}};
            x      <= {WIDTH{1'b0}};
            y      <= {WIDTH{1'b0}};
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                x    <= a;
                y    <= b;
                busy <= 1'b1;
            end else if (busy) begin
                if (y == {WIDTH{1'b0}}) begin
                    result <= x;
                    busy   <= 1'b0;
                    done   <= 1'b1;
                end else begin
                    x <= y;
                    y <= x % y;
                end
            end
        end
    end
endmodule
